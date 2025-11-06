import numpy as np
import joblib
import tflite_runtime.interpreter as tflite
import time
import serial
import sys # 오류 발생 시 프로그램 종료를 위해 추가

# -----------------------------
# AI 모델 및 변수 설정
# -----------------------------
# 사용할 모델 선택 (둘 중 하나 선택)
USE_MODEL = "multi_class" 
# USE_MODEL = "running" 

# 모델이 요구하는 시퀀스 길이 (학습 시 사용된 실제 값으로 반드시 수정해야 함)
TIME_STEPS = 50 

if USE_MODEL == "multi_class":
    MODEL_PATH = 'multi_class_model.tflite'
    SCALER_PATH = 'multi_class_scaler.joblib'
    # TODO: 모델의 실제 출력 순서에 맞는 라벨 리스트로 수정해야 합니다.
    LABELS = ['앉기', '서기', '걷기', '뛰기', '계단 오르기'] 
    
elif USE_MODEL == "running":
    MODEL_PATH = 'running_model.tflite'
    SCALER_PATH = 'running_scaler.joblib'
    # TODO: 모델의 실제 출력 순서에 맞는 라벨 리스트로 수정해야 합니다.
    LABELS = ['정상 움직임', '비정상 움직임'] 

# 로드될 객체 초기화
interpreter = None
scaler = None
data_buffer = []  # 실시간 데이터를 모을 버퍼 (TIME_STEPS 길이 유지)

try:
    # 1. TFLite 인터프리터 로드
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite 모델 로드 성공: {MODEL_PATH}")

    # 2. 전처리 스케일러 로드
    scaler = joblib.load(SCALER_PATH)
    print("스케일러 로드 성공")

except Exception as e:
    print(f"\n[오류] AI 모델/객체 로드 실패: {e}")
    print("모델 파일 경로, joblib 파일 경로, joblib 라이브러리 설치를 확인하세요.")
    sys.exit(1)

# -----------------------------
# 실시간 데이터 처리 함수 정의
# -----------------------------
def process_realtime_data(new_sensor_row):
    """
    새로운 센서 데이터를 버퍼에 추가하고, 시퀀스가 차면 모델 예측을 수행합니다.
    Args:
        new_sensor_row (np.array): STM에서 받은 센서 데이터 (예: shape=(12,))
    Returns:
        tuple or None: (score, class_name, confidence) 또는 None (버퍼 수집 중)
    """
    global data_buffer

    # 데이터 스케일링: joblib 스케일러는 (N, M) 형태의 입력을 기대하므로, reshape 필요
    try:
        scaled_data = scaler.transform(new_sensor_row.reshape(1, -1))
    except ValueError as ve:
        print(f"스케일링 오류: 데이터 형태를 확인하세요. {ve}")
        return None

    # 버퍼 관리: 시퀀스 길이 유지
    if len(data_buffer) >= TIME_STEPS:
        data_buffer.pop(0) # 가장 오래된 데이터 제거
    
    data_buffer.append(scaled_data[0])

    if len(data_buffer) < TIME_STEPS:
        return None # 시퀀스가 아직 채워지지 않음

    # ------------------
    # TFLite 모델 추론
    # ------------------
    # 입력 데이터 준비: (1, TIME_STEPS, 12) 형태로 변환 (np.float32 타입 중요)
    input_data = np.array(data_buffer, dtype=np.float32)
    input_data = input_data[np.newaxis, ...] 

    # 모델 입력 설정 및 실행
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # ------------------
    # 후처리
    # ------------------
    probabilities = output_data[0] # (N,) 형태의 확률 배열
    predicted_index = np.argmax(probabilities)
    
    # 예측 신뢰도 및 클래스 이름
    confidence = probabilities[predicted_index] * 100
    class_name = LABELS[predicted_index]

    # score는 신뢰도와 동일하게 설정 (모델이 별도의 정상도/이상도 점수를 출력하지 않는 경우)
    score = confidence

    return score, class_name, confidence

# -----------------------------
# 시리얼 포트 설정
# -----------------------------
SERIAL_PORT_STM = '/dev/ttyACM0'   # STM 데이터 입력
SERIAL_PORT_ESP = '/dev/ttyUSB0'   # ESP 결과 출력
BAUD_RATE = 115200

# -----------------------------
# 시리얼 연결
# -----------------------------
ser_stm = None
ser_esp = None

try:
    ser_stm = serial.Serial(SERIAL_PORT_STM, BAUD_RATE, timeout=1)
    print(f"STM 입력 시리얼 포트 열림: {SERIAL_PORT_STM} @ {BAUD_RATE}bps")
except Exception as e:
    print(f"STM 시리얼 연결 실패 (입력): {e}")

try:
    ser_esp = serial.Serial(SERIAL_PORT_ESP, BAUD_RATE, timeout=1)
    print(f"ESP 출력 시리얼 포트 열림: {SERIAL_PORT_ESP} @ {BAUD_RATE}bps")
except Exception as e:
    print(f"ESP 시리얼 연결 실패 (출력): {e}")

# -----------------------------
# 메인 루프
# -----------------------------
print("\n--- 실시간 예측 시작 ---")
FRAME_COUNT = 120 # 테스트용 프레임 수
try:
    for i in range(FRAME_COUNT): 
        new_sensor_row = None
        
        # STM에서 센서 데이터 읽기
        if ser_stm and ser_stm.in_waiting:
            line = ser_stm.readline().decode('latin-1').strip() # 인코딩 오류 방지를 위해 latin-1 사용
            try:
                # 콤마로 분리된 데이터를 정수 numpy 배열로 변환
                new_sensor_row = np.array([int(x) for x in line.split(',') if x.strip()], dtype=np.int32)
                # 데이터 길이가 12가 아닐 경우 건너뛰기
                if new_sensor_row.shape[0] != 12:
                    new_sensor_row = None
            except Exception as parse_e:
                # 데이터 파싱 오류
                print(f"데이터 파싱 오류: {line} -> {parse_e}")
                new_sensor_row = None
        
        # 시리얼 데이터가 없거나 파싱 오류 시 테스트용 랜덤 데이터 사용
        if new_sensor_row is None:
            # 테스트용 랜덤 데이터 (12개 특성)
            new_sensor_row = np.random.randint(-15000, 15000, size=12, dtype=np.int32)
        
        result = process_realtime_data(new_sensor_row)
        
        if result is not None:
            score, class_name, conf = result
            # ESP로 전송할 CSV 형식 문자열
            csv_line = f"{score:.2f},{class_name}\n"
            
            print(f"[프레임 {i+1}/{FRAME_COUNT}] 정상도(신뢰도): {score:.2f}%, 자세: {class_name} ({conf:.2f}%)")
            
            if ser_esp:
                ser_esp.write(csv_line.encode('utf-8'))
        else:
            print(f"[프레임 {i+1}/{FRAME_COUNT}] 데이터 수집 중 ({len(data_buffer)}/{TIME_STEPS})")
        
        time.sleep(0.02) # 50Hz (20ms) 주기로 실행

except KeyboardInterrupt:
    print("\n사용자에 의해 종료됨.")
except Exception as e:
    print(f"\n예기치 않은 오류 발생: {e}")

# -----------------------------
# 종료
# -----------------------------
if ser_stm and ser_stm.is_open:
    ser_stm.close()
if ser_esp and ser_esp.is_open:
    ser_esp.close()
print("\n시리얼 포트 종료")