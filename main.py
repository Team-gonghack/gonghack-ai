import numpy as np
import joblib
import tflite_runtime.interpreter as tflite
import time
import serial

# -----------------------------
# 1. 설정값
# -----------------------------
SCALER_PATH_SCORE = 'running_scaler.joblib'     # 정상도 모델 스케일러
TFLITE_PATH_SCORE = 'running_model.tflite'     # 정상도 모델

SCALER_PATH_CLASS = 'multi_class_scaler.joblib' # 자세 모델 스케일러
TFLITE_PATH_CLASS = 'multi_class_model.tflite' # 자세 모델

TIME_STEPS = 100

CLASS_NAMES = {0: "걷기 (Walk)", 1: "뛰기 (Run)", 2: "정지 (Stop)"}

# 시리얼 포트 설정 (Raspberry Pi 예시)
SERIAL_PORT = '/dev/ttyUSB0'  # PC에서는 COM3 등으로 변경
BAUD_RATE = 115200

# -----------------------------
# 2. 모델 및 스케일러 로드
# -----------------------------
print("스케일러 로드 중...")
scaler_score = joblib.load(SCALER_PATH_SCORE)
scaler_class = joblib.load(SCALER_PATH_CLASS)

print("TFLite 모델 로드 및 텐서 할당 중...")
interpreter_score = tflite.Interpreter(model_path=TFLITE_PATH_SCORE)
interpreter_score.allocate_tensors()
interpreter_class = tflite.Interpreter(model_path=TFLITE_PATH_CLASS)
interpreter_class.allocate_tensors()

input_details_score = interpreter_score.get_input_details()
output_details_score = interpreter_score.get_output_details()
input_details_class = interpreter_class.get_input_details()
output_details_class = interpreter_class.get_output_details()

print(f"정상도 모델 입력: {input_details_score[0]['shape']}, 출력: {output_details_score[0]['shape']}")
print(f"자세 모델 입력: {input_details_class[0]['shape']}, 출력: {output_details_class[0]['shape']}")

# -----------------------------
# 3. 데이터 버퍼
# -----------------------------
data_buffer = []

# -----------------------------
# 4. 실시간 처리 함수
# -----------------------------
def process_realtime_data(new_row):
    """
    새 센서 데이터 한 줄(12개)을 받아서
    - 정상도(0~100%) 계산
    - 자세 예측(3클래스)
    둘 다 준비되면 반환, 아니면 None
    """
    data_buffer.append(new_row)
    
    if len(data_buffer) > TIME_STEPS:
        data_buffer.pop(0)
    
    if len(data_buffer) == TIME_STEPS:
        # --- 정상도 예측 ---
        scaled_score = scaler_score.transform(np.array(data_buffer))
        input_score = np.array([scaled_score], dtype=np.float32)
        interpreter_score.set_tensor(input_details_score[0]['index'], input_score)
        interpreter_score.invoke()
        output_score = interpreter_score.get_tensor(output_details_score[0]['index'])
        abnormal_prob = output_score[0][0]
        normality_score = (1 - abnormal_prob) * 100
        
        # --- 자세 예측 ---
        scaled_class = scaler_class.transform(np.array(data_buffer))
        input_class = np.array([scaled_class], dtype=np.float32)
        interpreter_class.set_tensor(input_details_class[0]['index'], input_class)
        interpreter_class.invoke()
        output_class = interpreter_class.get_tensor(output_details_class[0]['index'])
        probabilities = output_class[0]
        predicted_index = np.argmax(probabilities)
        predicted_class_name = CLASS_NAMES[predicted_index]
        confidence = probabilities[predicted_index] * 100
        
        return normality_score, predicted_class_name, confidence
    else:
        return None

# -----------------------------
# 5. 시리얼 연결
# -----------------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"시리얼 포트 열림: {SERIAL_PORT} @ {BAUD_RATE}bps")
except Exception as e:
    print(f"시리얼 연결 실패: {e}")
    ser = None

# -----------------------------
# 6. 메인 루프
# -----------------------------
print("\n--- 실시간 예측 시작 ---")
for i in range(120):  # 테스트용 120 프레임
    # 실제로는 센서에서 읽은 값
    new_sensor_row = np.random.randint(-15000, 15000, size=12)
    
    result = process_realtime_data(new_sensor_row)
    
    if result is not None:
        score, class_name, conf = result
        csv_line = f"{score:.2f},{class_name}\n"
        
        print(f"[프레임 {i}] 정상도: {score:.2f}%, 자세: {class_name} ({conf:.2f}%)")
        
        if ser:
            ser.write(csv_line.encode())
    else:
        print(f"[프레임 {i}] 데이터 수집 중 ({len(data_buffer)}/{TIME_STEPS})")
    
    time.sleep(0.02)

# -----------------------------
# 7. 종료
# -----------------------------
if ser:
    ser.close()
    print("시리얼 포트 종료")
