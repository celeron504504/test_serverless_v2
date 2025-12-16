# CUDA base image (runtime only, lightweight)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install Python + system deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first (better caching)
COPY requirements.txt .

# Install CUDA-enabled PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run handler
CMD ["python", "handler.py"]
