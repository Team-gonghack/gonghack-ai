# 라즈베리파이용 Python 3.11 기반 이미지 (arm32v7)
FROM --platform=linux/arm/v7 python:3.11-slim

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y build-essential python3-dev libatlas-base-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 필요한 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY realtime_predict_serial.py .
COPY running_scaler.joblib .
COPY running_model.tflite .
COPY multi_class_scaler.joblib .
COPY multi_class_model.tflite .

# 실행
CMD ["python", "realtime_predict_serial.py"]
