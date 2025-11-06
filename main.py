import numpy as np
import joblib
import tflite_runtime.interpreter as tflite
import time

# --- 1. 설정값 ---
SCALER_PATH = 'running_scaler.joblib'    # Colab에서 가져온 스케일러
TFLITE_PATH = 'running_model.tflite' # Colab에서 가져온 모델
TIME_STEPS = 100                     # 학습 때 사용한 값 (100)

# --- 2. 모델 및 스케일러 로드 ---
print("스케일러 로드 중...")
scaler = joblib.load(SCALER_PATH)

print("TFLite 모델 로드 및 텐서 할당 중...")
interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors() # TFLite 모델 메모리 할당

# TFLite 모델의 입력/출력 세부 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  - 모델 입력 Shape: {input_details[0]['shape']}") # [1, 100, 12] 여야 함

# --- 3. 실시간 처리를 위한 데이터 버퍼 ---
# 센서 데이터를 100개 모으기 위한 리스트(큐)
data_buffer = []

def process_realtime_data(new_row_data):
    """
    센서에서 방금 읽은 1줄(12개 값)의 데이터를 처리하는 함수.
    100개가 모이면 '정상도'를 반환, 아니면 None을 반환.
    """
    
    # 1. 버퍼에 새 데이터 추가
    data_buffer.append(new_row_data)
    
    # 2. 버퍼 크기 관리 (가장 오래된 데이터 삭제)
    if len(data_buffer) > TIME_STEPS:
        data_buffer.pop(0)
    
    # 3. 데이터가 100개가 모였는지 확인
    if len(data_buffer) == TIME_STEPS:
        # (1) 스케일링: scaler는 2D 배열을 기대 (100, 12)
        scaled_data = scaler.transform(np.array(data_buffer))
        
        # (2) TFLite 입력 형식 변환 (1, 100, 12) + float32
        input_tensor = np.array([scaled_data], dtype=np.float32)
        
        # (3) 모델 예측 실행
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        # (4) 결과 가져오기 (모델은 '비정상(1)'일 확률을 반환)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        abnormal_prob = output_data[0][0] # 0.0 ~ 1.0
        
        # (5) '정상도'로 변환
        normality_score = (1 - abnormal_prob) * 100
        return normality_score
    else:
        # 아직 100개 안 모임
        return None

# --- 4. 메인 루프 (시뮬레이션) ---
# (실제로는 이 부분을 RPI의 MPU6050 센서 데이터 읽는 코드로 대체해야 함)
print("\n--- 실시간 예측 시뮬레이션 시작 ---")
print(f"(센서 데이터 {TIME_STEPS}개가 모여야 첫 예측이 시작됩니다)")

# MPU6050에서 12개 컬럼 데이터를 1줄씩 읽어온다고 가정
# (예: new_row = read_mpu_sensors())
# 여기서는 테스트용으로 numpy 랜덤 데이터를 1줄씩 생성
for i in range(120): # 120 프레임 시뮬레이션
    
    # [시뮬레이션] 센서에서 (1, 12) 모양의 새 데이터를 읽었다고 가정
    # (주의: 실제 센서 raw 값 범위에 맞춰야 함)
    # new_sensor_row = mpu.get_all_data() # 예시
 
   new_sensor_row = np.random.randint(-15000, 15000, size=12) 
    
    # 새 데이터를 처리 함수에 전달
    score = process_realtime_data(new_sensor_row)
    
    if score is not None:
        # 100개가 모여서 점수가 나옴
        print(f" [프레임 {i}] 예측 완료! ➡️ 정상도: {score:.2f}%")
    else:
        # 100개 모으는 중
        print(f" [프레임 {i}] 데이터 수집 중... ({len(data_buffer)}/{TIME_STEPS})")
        
    time.sleep(0.02) # 실제 센서 딜레이처럼 0.02초 대기