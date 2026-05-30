
import streamlit as st
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bone Fracture Detection", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Preprocessing: remove glare / light artifacts from xray
# ─────────────────────────────────────────────────────────────────────────────
def remove_glare(img_bgr):
    """
    Detect homogeneous bright blobs (light-source glare on the xray film)
    and inpaint over them, then apply CLAHE to boost bone contrast.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.blur(gray, (31, 31))
    blur_sq = cv2.blur(gray ** 2, (31, 31))
    local_std = np.sqrt(np.maximum(blur_sq - blur ** 2, 0))

    # Glare = bright AND uniform (low local std-dev)
    glare_mask = ((gray > 180) & (local_std < 15)).astype(np.uint8)
    glare_mask = cv2.dilate(glare_mask, np.ones((15, 15), np.uint8))

    inpainted = cv2.inpaint(img_bgr, glare_mask, 21, cv2.INPAINT_TELEA)
    gray_inp  = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = cv2.merge([clahe.apply(gray_inp)] * 3)
    return enhanced, glare_mask

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – EigenCAM on multiple backbone layers
# ─────────────────────────────────────────────────────────────────────────────
def compute_eigencam(img_bgr, torch_model, target_layers=(6, 12, 15, 18)):
    oh, ow = img_bgr.shape[:2]
    lb      = LetterBox(new_shape=(1024, 1024))
    resized = lb(image=img_bgr)
    inp     = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    hooks, handles = {}, []
    def make_hook(k):
        def fn(m, i, out): hooks[k] = out.detach()
        return fn
    for idx in target_layers:
        try: handles.append(torch_model.model[idx].register_forward_hook(make_hook(idx)))
        except Exception: pass

    with torch.no_grad(): torch_model(inp)
    for hh in handles: hh.remove()

    scale = 1024 / max(oh, ow)
    new_h, new_w = int(oh * scale), int(ow * scale)
    pad_x, pad_y = (1024 - new_w) // 2, (1024 - new_h) // 2

    combined = np.zeros((1024, 1024), dtype=np.float32)
    for feat in hooks.values():
        f = feat[0].cpu().numpy()
        C, H, W = f.shape
        flat     = f.reshape(C, -1)
        centered = flat - flat.mean(axis=1, keepdims=True)
        v = np.ones(C) / np.sqrt(C)
        cov = centered @ centered.T / (H * W)
        for _ in range(30):
            v = cov @ v
            nrm = np.linalg.norm(v)
            if nrm < 1e-10: break
            v /= nrm
        cam = np.maximum((v @ flat).reshape(H, W), 0)
        if cam.max() > 1e-8: cam /= cam.max()
        combined += cv2.resize(cam, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    cam_crop = combined[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    cam_orig = cv2.resize(cam_crop, (ow, oh), interpolation=cv2.INTER_CUBIC)
    cam_orig = np.maximum(cam_orig, 0)
    if cam_orig.max() > 1e-8: cam_orig /= cam_orig.max()
    return cam_orig

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Build bone mask (exclude dark background + glare artifacts)
# ─────────────────────────────────────────────────────────────────────────────
def build_bone_mask(gray, min_area_frac=0.03):
    oh, ow = gray.shape
    candidate = ((gray > 40) & (gray < 225)).astype(np.uint8) * 255
    closed    = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    mask = np.zeros((oh, ow), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > oh * ow * min_area_frac:
            mask[labels == i] = 255
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Detect fracture peaks and draw tight boxes
# ─────────────────────────────────────────────────────────────────────────────
def get_fracture_boxes(cam, gray, bone_mask, cam_threshold=0.70,
                       peak_search_size=60, max_radius=120, min_radius=25):
    oh, ow = cam.shape
    smooth = gaussian_filter(cam, sigma=max(oh, ow) * 0.025)
    smooth /= smooth.max() + 1e-8

    # Local maxima
    local_max = smooth == maximum_filter(smooth, size=peak_search_size)
    peaks_y, peaks_x = np.where(local_max & (smooth > cam_threshold))

    boxes = []
    for py, px in zip(peaks_y, peaks_x):
        score = float(smooth[py, px])

        # ── Bone filter: reject peaks on background or pure glare ────────────
        r = 30
        y1r, y2r = max(0, py - r), min(oh, py + r)
        x1r, x2r = max(0, px - r), min(ow, px + r)
        patch_bone      = bone_mask[y1r:y2r, x1r:x2r].mean() / 255
        patch_intensity = float(gray[y1r:y2r, x1r:x2r].mean())
        if patch_bone < 0.30 or patch_intensity < 30 or patch_intensity > 228:
            continue  # skip artifact / background

        # ── Radius from CAM half-power width ─────────────────────────────────
        sub_r = 80
        sub   = smooth[max(0, py - sub_r):min(oh, py + sub_r),
                       max(0, px - sub_r):min(ow, px + sub_r)]
        half  = sub >= score * 0.55
        radius = int(np.sqrt(half.sum() / np.pi) * 1.3)
        radius = max(min_radius, min(max_radius, radius))

        boxes.append({
            "center": (px, py),
            "box":    (max(0, px - radius), max(0, py - radius),
                       min(ow, px + radius), min(oh, py + radius)),
            "score":  score,
            "radius": radius,
        })

    # ── Merge overlapping boxes ───────────────────────────────────────────────
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        used   = [False] * len(boxes)
        new_boxes = []
        for i in range(len(boxes)):
            if used[i]: continue
            grp = [boxes[i]]
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                ax1,ay1,ax2,ay2 = boxes[i]["box"]
                bx1,by1,bx2,by2 = boxes[j]["box"]
                # Overlap?
                if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                    grp.append(boxes[j])
                    used[j] = True
                    merged  = True
            mx1 = min(b["box"][0] for b in grp)
            my1 = min(b["box"][1] for b in grp)
            mx2 = max(b["box"][2] for b in grp)
            my2 = max(b["box"][3] for b in grp)
            new_boxes.append({
                "center": ((mx1+mx2)//2, (my1+my2)//2),
                "box":    (mx1,my1,mx2,my2),
                "score":  max(b["score"] for b in grp),
                "radius": max(b["radius"] for b in grp),
            })
        boxes = new_boxes

    boxes.sort(key=lambda b: -b["score"])
    return boxes, smooth

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Draw results on image
# ─────────────────────────────────────────────────────────────────────────────
def draw_results(img_bgr, boxes, smooth, show_heatmap=True, heatmap_alpha=0.30):
    out = img_bgr.copy()
    if show_heatmap:
        cam_u8    = (smooth * 255).astype(np.uint8)
        cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        out       = cv2.addWeighted(out, 1 - heatmap_alpha, cam_color, heatmap_alpha, 0)

    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        score           = b["score"]
        color           = (0, 255, 80)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"Fracture {score:.0%}"
        font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, th), bl = cv2.getTextSize(label, font, fscale, fthick)
        ly = max(y1 - 6, th + 6)
        cv2.rectangle(out, (x1, ly - th - bl - 2), (x1 + tw + 6, ly + 2), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 3, ly - bl), font, fscale, (0, 0, 0), fthick, cv2.LINE_AA)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("Bone Fracture Detection using AI")
st.markdown(
    "Just upload your Xray and get the results instantly .... Because we understand , every second matters in diagnosis!!"
)

with st.sidebar:
    st.header("⚙️ Settings")
    cam_thresh = st.slider(
        "Detection sensitivity", 0.50, 0.90, 0.70, 0.05,
        help="Lower = more regions detected. Raise if you see false positives."
    )
    show_heatmap  = st.checkbox("Show attention heatmap", value=True)
    heatmap_alpha = st.slider("Heatmap opacity", 0.10, 0.60, 0.30, 0.05,
                              disabled=not show_heatmap)
    st.divider()
    

uploaded = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode image.")
        st.stop()

    with st.spinner("Processing…"):
        # Preprocess
        enhanced, glare_mask = remove_glare(img_bgr)
        gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bone_mask = build_bone_mask(gray_orig)

        # EigenCAM on glare-free image
        torch_model = model.model
        torch_model.eval()
        cam = compute_eigencam(enhanced, torch_model)

        # Detect fracture locations
        boxes, smooth = get_fracture_boxes(
            cam, gray_orig, bone_mask, cam_threshold=cam_thresh
        )

    # ── Display ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with col1:
        st.subheader("Original")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        if not boxes:
            st.success("✅ No fracture detected above threshold.")
            if show_heatmap:
                cam_color = cv2.applyColorMap((smooth*255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay   = cv2.addWeighted(img_bgr, 0.70, cam_color, 0.30, 0)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.image(img_rgb, use_container_width=True)
        else:
            result = draw_results(img_bgr, boxes, smooth, show_heatmap, heatmap_alpha)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    if boxes:
        st.subheader(f"🔍 {len(boxes)} Fracture Region(s) Detected")
        rows = []
        for i, b in enumerate(boxes):
            x1,y1,x2,y2 = b["box"]
            rows.append({
                "#":           i + 1,
                "Confidence":  f"{b['score']:.0%}",
                "Location":    f"({x1},{y1}) → ({x2},{y2})",
                "Size (px)":   f"{x2-x1} × {y2-y1}",
            })
        st.table(rows)
    else:
        st.info("Try lowering **Detection sensitivity** in the sidebar.")
