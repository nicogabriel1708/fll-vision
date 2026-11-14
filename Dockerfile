# ==============================================================================
# FLL-Vision Production Dockerfile (Single-Container Backend + Frontend)
# Target Platform: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip, linux/arm64)
# ==============================================================================
FROM nvcr.io/nvidia/pytorch:26.10-py3

LABEL maintainer="FLL-Vision Team"
LABEL platform="NVIDIA DGX Spark (GB10 Grace Blackwell / linux/arm64)"
LABEL description="Complete FLL Robot Game Livestream, Dual Auto-Zoom & TensorRT Tracker in a Single Container"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Target NVIDIA GB10 Blackwell GPU Architecture (SM 12.0 / 12.1) & Grace ARM64 CPU cores
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"
# Target NVIDIA GB10 Blackwell GPU Architecture (SM 12.0 / 12.1) & Grace ARM64 CPU cores
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"
ENV CUDA_MODULE_LOADING=LAZY
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
ENV YOLO_AUTOINSTALL=False

WORKDIR /workspace

# Install system libraries for OpenCV, hardware video decoding, FFmpeg, and SQLite3
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    sqlite3 \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages without overwriting NGC's hardware-optimized torch & tensorrt
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore \
    --extra-index-url https://pypi.ngc.nvidia.com \
    opencv-python-headless \
    polars \
    pyarrow \
    matplotlib \
    pillow \
    psutil \
    tqdm \
    requests \
    py-cpuinfo \
    ultralytics-thop \
    ultralytics-platform \
    pyyaml \
    pydantic \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    websockets \
    onnx \
    onnxslim \
    onnx-graphsurgeon \
    onnxruntime \
    lief \
    polygraphy \
    "nvidia-modelopt[onnx]" \
    scipy && \
    pip install --no-cache-dir --root-user-action=ignore --no-deps ultralytics lapx filterpy

# Create required directory structure (including empty training_videos directory)
RUN mkdir -p /workspace/config \
             /workspace/src \
             /workspace/scripts \
             /workspace/frontend/templates \
             /workspace/frontend/static \
             /workspace/models \
             /workspace/data/training_videos \
             /workspace/data/videos \
             /workspace/data/outputs \
             /workspace/data/db

# Copy configuration, backend source, frontend templates + static assets, and scripts
COPY config/ /workspace/config/
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY frontend/ /workspace/frontend/
COPY entrypoint.sh /workspace/entrypoint.sh
COPY setup.sh /workspace/setup.sh
COPY run_all.sh /workspace/run_all.sh

RUN chmod +x /workspace/entrypoint.sh /workspace/setup.sh /workspace/run_all.sh /workspace/scripts/setup_db.py

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/health > /dev/null || exit 1

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["python3", "-m", "src.main", "--config", "/workspace/config/config.yaml"]
