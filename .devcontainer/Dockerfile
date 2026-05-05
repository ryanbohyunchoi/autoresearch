# Python 3.11 slim base — no GPU needed for embedding-based classification locally
FROM python:3.11-slim

# System basics
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user "dev" matching host UID to avoid permission issues
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if getent passwd $USER_UID > /dev/null 2>&1; then \
        userdel -r $(getent passwd $USER_UID | cut -d: -f1); \
    fi

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && mkdir -p /etc/sudoers.d \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
