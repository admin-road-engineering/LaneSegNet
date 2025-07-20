# Use an NVIDIA CUDA runtime base image with CUDA 12.1 support
# Tag for Ubuntu 22.04. Other OS versions are available.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHON_VERSION_MAJOR=3
ENV PYTHON_VERSION_MINOR=11
# Use a stable Python version
ENV PYTHON_VERSION=3.11

# Install system dependencies including Python from deadsnakes PPA, pip, git, build tools, and OpenCV libs
# Added cmake, autoconf, automake, libtool for potential from-source builds
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python${PYTHON_VERSION}-venv \
    # Installs pip for the system's default python, we will upgrade/use it for 3.11
    python3-pip \
    git \
    wget \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    # System dependencies for OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to the specific version we installed from deadsnakes
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}

# Upgrade pip for the selected Python version and install essential Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set the working directory in the container
WORKDIR /app

# --- Install PyTorch ---
# Ensure PyTorch matches the CUDA version in the base image (12.1)
# For PyTorch 2.1.2 and CUDA 12.1 (compatible with mmcv==2.1.0 and mmsegmentation==1.2.2)
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# --- Install MMCV ---
# Force compilation from source
# Set environment variables that mmcv build might use
ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1
# Explicitly set CUDA architectures to compile for.
# See https://developer.nvidia.com/cuda-gpus#compute for architecture codes
# RTX 3060 is Ampere (8.6)
ENV TORCH_CUDA_ARCH_LIST="8.6"

# For PyTorch 2.1.x, mmcv 2.1.0 is appropriate.
# Use --no-binary mmcv to force compilation only for mmcv. Build can take time.
RUN python3 -m pip install --no-cache-dir --no-binary mmcv "mmcv==2.1.0"

# --- Install MMSegmentation (stable version) ---
RUN python3 -m pip install --no-cache-dir "mmsegmentation==1.2.2"

# Copy the application code into the container
COPY ./app /app/app
COPY test_cuda_mmcv.py /app/test_cuda_mmcv.py

# Copy model configuration and weights
COPY ./configs /app/configs
COPY ./weights /app/weights

# Copy requirements (ensure mmcv, torch, mmseg are NOT listed or commented out)
COPY requirements.txt .
# Copy model config and checkpoint if they exist locally and are needed
# COPY ./models /app/models # Commented out: Models likely loaded at runtime, directory may not exist at build time

# Install remaining dependencies from requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8010

# Define the command to run the application using Uvicorn
# Use python3 to ensure the correct version is used
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"] 