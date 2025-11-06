import numpy as np
import joblib
import tensorflow as tf   # TensorFlow TFLite
import time
import serial
import sys

# --- 1. 파일 경로 및 설정 ---
TIME_STEPS = 100

SCALER_MODEL1_PATH = 'binary_scaler_compatible.joblib'
TFLITE_MODEL1_PATH = 'binary_model_compatible.tflite'
SCALER_MODEL2_PATH = 'multi_class_scaler_compatible.joblib'
TFLITE_MODEL2_PATH = 'multi_class_model_compatible.tflite'

CLASS_NAMES_MODEL2 = {0: "걷기", 1: "뛰기", 2: "정지"}

# --- 2. 모델 및 스케일러 로드 ---
try:
    scaler_model1 = joblib.load(SCALER_MODEL1_PATH)
    scaler_model2 = joblib.load(SCALER_MODEL2_PATH)

    interpreter_model1 = tf.lite.Interpreter(model_path=TFLITE_MODEL1_PATH)
    interpreter_model1.allocate_tensors()
    input_details_m1 = interpreter_model1.get_input_details()
    output_details_m1 = interpreter_model1.get_output_details()

    interpreter_model2 = tf.lite.Interpreter(model_path=TFLITE_MODEL2_PATH)
    interpreter_model2.allocate_tensors()
    input_details_m2 = interpreter_model2.get_input_details()
    output_details_m2 = interpreter_model2.get_output_details()

except Exception as e:
    print(f"모델/스케일러 로드 실패: {e}")
    sys.exit(1)

# --- 3. 시리얼 포트 설정 ---
SERIAL_PORT_STM = '/dev/ttyACM0'
SERIAL_PORT_ESP = '/dev/ttyUSB0'
BAUD_RATE = 115200

try:
    ser_stm = serial.Serial(SERIAL_PORT_STM, BAUD_RATE, timeout=1)
    print(f"STM 포트 연결: {SERIAL_PORT_STM}")
except Exception as e:
    print(f"STM 시리얼 연결 실패: {e}")
    ser_stm = None

try:
    ser_esp = serial.Serial(SERIAL_PORT_ESP, BAUD_RATE, timeout=1)
    print(f"ESP 포트 연결: {SERIAL_PORT_ESP}")
except Exception as e:
    print(f"ESP 시리얼 연결 실패: {e}")
    ser_esp = None

# --- 4. 데이터 버퍼 ---
data_buffer = []

def process_realtime_data(new_row_data):
    data_buffer.append(new_row_data)
    print(f"[DEBUG] 새 센서 데이터 추가: {new_row_data}")
    print(f"[DEBUG] 버퍼 크기: {len(data_buffer)}/{TIME_STEPS}")

    if len(data_buffer) > TIME_STEPS:
        removed = data_buffer.pop(0)
        print(f"[DEBUG] 버퍼에서 제거된 오래된 데이터: {removed}")

    if len(data_buffer) == TIME_STEPS:
        buffer_array = np.array(data_buffer)
        print(f"[DEBUG] 모델 입력 배열 shape: {buffer_array.shape}")

        # 모델1: 정상/비정상
        scaled_m1 = scaler_model1.transform(buffer_array)
        input_tensor_m1 = np.array([scaled_m1], dtype=np.float32)
        interpreter_model1.set_tensor(input_details_m1[0]['index'], input_tensor_m1)
        interpreter_model1.invoke()
        output_data_m1 = interpreter_model1.get_tensor(output_details_m1[0]['index'])
        abnormal_prob = output_data_m1[0][0]
        normality_score = int((1 - abnormal_prob) * 100)
        print(f"[DEBUG] 모델1 출력(비정상 확률): {abnormal_prob:.4f} → 정상도: {normality_score}%")

        # 모델2: 동작 분류
        scaled_m2 = scaler_model2.transform(buffer_array)
        input_tensor_m2 = np.array([scaled_m2], dtype=np.float32)
        interpreter_model2.set_tensor(input_details_m2[0]['index'], input_tensor_m2)
        interpreter_model2.invoke()
        output_data_m2 = interpreter_model2.get_tensor(output_details_m2[0]['index'])
        probabilities_m2 = output_data_m2[0]
        pred_index = np.argmax(probabilities_m2)
        class_name = CLASS_NAMES_MODEL2[pred_index]
        print(f"[DEBUG] 모델2 출력 확률: {probabilities_m2} → 예측: {class_name}")

        return normality_score, class_name
    else:
        return None

# --- 5. 메인 루프 ---
print("\n--- 실시간 예측 + STM → ESP32 시리얼 통신 시작 ---")

try:
    while True:
        new_row = None

        if ser_stm and ser_stm.in_waiting:
            line = ser_stm.readline().decode('latin-1').strip()
            print(f"[DEBUG] STM 수신 라인: {line}")
            try:
                values = [int(x) for x in line.split(',') if x.strip()]
                if len(values) == 12:
                    new_row = np.array(values, dtype=np.int32)
            except Exception as e:
                print(f"[ERROR] STM 데이터 파싱 실패: {line} -> {e}")

        if new_row is None:
            continue

        result = process_realtime_data(new_row)

        if result is not None:
            normality_score, class_name = result
            print(f"[RESULT] 정상도: {normality_score}% | 동작: {class_name}")

            if ser_esp:
                csv_line = f"{normality_score},{class_name}\n"
                ser_esp.write(csv_line.encode('utf-8'))
                print(f"[ESP32 전송] {csv_line.strip()}")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("종료")

finally:
    if ser_stm and ser_stm.is_open:
        ser_stm.close()
    if ser_esp and ser_esp.is_open:
        ser_esp.close()
    print("시리얼 포트 종료")
