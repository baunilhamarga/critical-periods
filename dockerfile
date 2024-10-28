# Dockerfile for your specific environment
FROM nvidia/cuda:12.3.0-cudnn8-runtime-ubuntu20.04

# Set environment variables to suppress interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release

# Install Clang 17.0.6
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 17
RUN apt-get install -y clang-17 libclang-17-dev

# Update the alternatives to use Clang 17 by default
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100

# Install Python 3.12.5 manually
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Set the Python3.12 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install pip and other tools
RUN python3 -m pip install --upgrade pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# Set up the CUDA and cuDNN environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Set up the working directory
WORKDIR /app

# Copy all files and folders from the current directory to the container
COPY . /app

# Optional: if you have a specific entry point
# ENTRYPOINT [ "python", "your_script.py" ]

# Final cleanup of cache to keep image small
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
