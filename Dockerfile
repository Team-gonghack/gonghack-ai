# Raspberry Pi 64-bit 기반 이미지 (Debian Bookworm)
FROM --platform=linux/arm64 python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3-dev \
    python3-venv \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    libreadline-dev \
    libncurses5-dev \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사
COPY requirements.txt .

# pip 업그레이드 & requirements 설치
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# tflite-runtime 설치 (Python 3.11 + ARM64용)
RUN pip install --no-cache-dir \
    https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.14.0-cp311-cp311-linux_aarch64.whl

# 포트 개방 예시 (모든 포트 열기)
EXPOSE 5000 8000 8080 8888 1234  # 필요에 맞게 추가

# 컨테이너 시작 시 기본 셸
CMD ["/bin/bash"]
