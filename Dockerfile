FROM python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libxcb1 \
    libglx-mesa0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

ENV MAKEFLAGS="-j1"
ENV PIP_NO_CACHE_DIR=1
ENV TMPDIR=/tmp

# Install heavy packages first separately, then the rest
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir streamlit opencv-python-headless ultralytics scipy

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "uvicorn app2.py:app --host 0.0.0.0 --port ${PORT:-10000}"]