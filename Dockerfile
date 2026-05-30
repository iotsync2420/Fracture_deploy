FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libxcb1 \
    libglx-mesa0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV MAKEFLAGS="-j1"
ENV PIP_NO_CACHE_DIR=1
ENV TMPDIR=/tmp

RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir streamlit opencv-python-headless ultralytics scipy

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user . .

EXPOSE 7860

# Multi-line list strings ko streamline execution me single standard line command bana diya
CMD ["streamlit", "run", "app2.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
