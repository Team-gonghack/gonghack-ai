import numpy as np
import joblib
import tflite_runtime.interpreter as tflite
import time
import serial

# -----------------------------
# 시리얼 포트 설정
# -----------------------------
SERIAL_PORT_STM = '/dev/ttyACM0'   # STM 데이터 입력
SERIAL_PORT_ESP = '/dev/ttyUSB0'   # ESP 결과 출력
BAUD_RATE = 115200

# -----------------------------
# 시리얼 연결
# -----------------------------
try:
    ser_stm = serial.Serial(SERIAL_PORT_STM, BAUD_RATE, timeout=1)
    print(f"STM 입력 시리얼 포트 열림: {SERIAL_PORT_STM} @ {BAUD_RATE}bps")
except Exception as e:
    print(f"STM 시리얼 연결 실패: {e}")
    ser_stm = None

try:
    ser_esp = serial.Serial(SERIAL_PORT_ESP, BAUD_RATE, timeout=1)
    print(f"ESP 출력 시리얼 포트 열림: {SERIAL_PORT_ESP} @ {BAUD_RATE}bps")
except Exception as e:
    print(f"ESP 시리얼 연결 실패: {e}")
    ser_esp = None

# -----------------------------
# 메인 루프
# -----------------------------
print("\n--- 실시간 예측 시작 ---")
for i in range(120):  # 테스트용 120 프레임
    # STM에서 센서 데이터 읽기
    if ser_stm and ser_stm.in_waiting:
        line = ser_stm.readline().decode().strip()
        try:
            new_sensor_row = np.array([int(x) for x in line.split(',')])
        except:
            continue
    else:
        # 테스트용 랜덤 데이터
        new_sensor_row = np.random.randint(-15000, 15000, size=12)
    
    result = process_realtime_data(new_sensor_row)
    
    if result is not None:
        score, class_name, conf = result
        csv_line = f"{score:.2f},{class_name}\n"
        
        print(f"[프레임 {i}] 정상도: {score:.2f}%, 자세: {class_name} ({conf:.2f}%)")
        
        if ser_esp:
            ser_esp.write(csv_line.encode())
    else:
        print(f"[프레임 {i}] 데이터 수집 중 ({len(data_buffer)}/{TIME_STEPS})")
    
    time.sleep(0.02)

# -----------------------------
# 종료
# -----------------------------
if ser_stm:
    ser_stm.close()
if ser_esp:
    ser_esp.close()
print("시리얼 포트 종료")
