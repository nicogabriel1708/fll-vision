# Stage 1: Build OpenCV with CUDA
FROM nvidia/cuda:13.0.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    python3-dev python3-pip python3-numpy \
    libopenblas-dev liblapack-dev libv4l-dev wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Clone OpenCV and contrib modules
WORKDIR /opt
RUN git clone --branch 4.8.0 https://github.com/opencv/opencv.git && \
    git clone --branch 4.8.0 https://github.com/opencv/opencv_contrib.git

# Build OpenCV
WORKDIR /opt/opencv/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN="8.6" \
    -DWITH_CUDNN=ON \
    -DOPENCV_DNN_CUDA=ON \
    -DWITH_TBB=ON \
    -DWITH_OPENGL=ON \
    -DWITH_V4L=ON \
    -DWITH_GSTREAMER=ON \
    -DBUILD_opencv_python3=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Stage 2: Runtime
FROM nvidia/cuda:13.0.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip libv4l-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy OpenCV from builder
COPY --from=builder /usr/local /usr/local

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run
CMD ["python3", "run.py"]

