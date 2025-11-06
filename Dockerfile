# 라즈베리파이용 Python 3.11 기반 이미지 (arm32v7)
FROM --platform=linux/arm/v7 python:3.9-slim

# 작업 디렉토리
WORKDIR /app

# 애플리케이션 코드 복사
COPY main.py .
COPY running_scaler.joblib .
COPY running_model.tflite .
COPY multi_class_scaler.joblib .
COPY multi_class_model.tflite .

RUN pip install numpy==1.25.2 joblib tflite-runtime pyserial

# 실행
CMD ["python", "main.py"]
