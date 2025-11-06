import serial
import csv
import time
import sys
import os # os.path.exists 추가

### TODO: 이 부분을 실제 COM 포트 번호로 수정하세요 ###
SERIAL_PORT = 'COM14'  # 2개의 IMU 데이터가 모두 들어오는 포트
BAUD_RATE = 115200
##################################################

def create_csv_header(prefixes):
    """ ['label1', 'label2', ...] 리스트를 받아 CSV 헤더를 생성합니다. """
    header = []
    # 각 IMU는 6축 (가속도 3 + 자이로 3) 데이터를 가짐
    suffixes = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
    
    for prefix in prefixes:
        for suffix in suffixes:
            header.append(f'{prefix}_{suffix}')
    return header

def parse_mpu_line(line):
    """ 'MPU2,ax,ay,az,x,y,z' 같은 줄을 파싱하여 [2, ['ax','ay','az','x','y','z']]를 반환 """
    try:
        parts = line.split(',')
        # MPU ID 1개 + 데이터 6개 = 총 7개 파트
        if len(parts) != 7:
            return None, None # 데이터 형식이 맞지 않음
            
        mpu_id_str = parts[0] # "MPU2"
        mpu_id = int(mpu_id_str[3:]) # "2"
        
        values = parts[1:] # 6개의 데이터 값 리스트
        return mpu_id, values
        
    except Exception as e:
        print(f"[Parser Error] 파싱 오류: {line}, 오류: {e}")
        return None, None

def start_logging(port, baud, filename, header, imu_labels, imu_ids):
    """
    [메인 로깅 함수]
    하나의 시리얼 포트에서 'imu_ids' 목록의 IMU 데이터가
    모두 수집되면 CSV 파일에 1줄로 저장합니다.
    """
    
    expected_num_imus = len(imu_ids)
    first_id = min(imu_ids) # 이 샘플의 시작 ID (e.g., 2)
    
    print(f"'{port}' 포트를 {baud}bps로 여는 중...")
    print(f"수집 대상 IMU (총 {expected_num_imus}개): {imu_labels}")
    print(f"수집 ID (순서대로): {imu_ids}")
    
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                
                print(f"'{filename}' 파일 생성 완료. 로깅을 시작합니다.")
                print("데이터 수집을 중지하려면 Ctrl+C 를 누르세요.")
                
                csv_writer = csv.writer(f)
                
                # 1. CSV 파일에 헤더(열 이름)를 씁니다.
                csv_writer.writerow(header)
                
                line_count = 0
                sample_buffer = {} # MPU 데이터를 임시 저장할 딕셔너리
                
                while True:
                    try:
                        serial_line = ser.readline()
                        
                        if not serial_line:
                            continue
                            
                        decoded_line = serial_line.decode('utf-8').strip()
                        if not decoded_line:
                            continue
                            
                        # 2. 데이터 파싱
                        mpu_id, values = parse_mpu_line(decoded_line)
                        
                        if mpu_id is None:
                            # print(f"경고: 데이터 형식 오류. 건너뜁니다: {decoded_line}")
                            continue
                        
                        # 3. 데이터 수집
                        # 수집 대상 ID가 아니면(예: MPU1, MPU4) 무시
                        if mpu_id not in imu_ids:
                            continue

                        # 첫 번째 ID(MPU2)를 받으면 버퍼 초기화
                        if mpu_id == first_id:
                            sample_buffer = {} # 버퍼 비우기
                            sample_buffer[mpu_id] = values
                        
                        # 첫 번째 ID가 수신된 상태에서, 순차적으로 다음 ID 수집
                        # (예: MPU3은 MPU2가 있어야 수집됨)
                        elif (mpu_id - 1) in sample_buffer:
                            sample_buffer[mpu_id] = values
                        
                        # 4. 데이터 세트 완성 확인
                        # (버퍼에 쌓인 데이터 개수가 필요한 IMU 개수와 같으면)
                        if len(sample_buffer) == expected_num_imus:
                            
                            # ID 순서대로 [2, 3] 데이터를 1줄의 리스트로 합치기
                            csv_row = []
                            for i in sorted(imu_ids):
                                csv_row.extend(sample_buffer[i])
                            
                            # 5. CSV에 저장
                            csv_writer.writerow(csv_row)
                            line_count += 1
                            
                            if line_count % 100 == 0:
                                print(f"[{line_count} 샘플] 저장 완료.")
                            
                            # 버퍼 비우기 (다음 MPU2를 기다림)
                            sample_buffer = {} 
                    
                    except UnicodeDecodeError:
                        print("경고: 데이터 디코딩 오류. (통신 오류 가능성)")
                    except KeyboardInterrupt:
                        print(f"\n데이터 수집 중지. 총 {line_count} 샘플이 '{filename}'에 저장되었습니다.")
                        break
                        
    except serial.SerialException:
        print(f"오류: '{port}' 포트를 찾을 수 없거나 열 수 없습니다.")
        print("1. 장치가 PC에 연결되었는지 확인하세요.")
        print(f"2. {SERIAL_PORT} 변수가 올바른 포트 번호인지 확인하세요.")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    
    label = input("이 데이터의 '라벨(label)'을 입력하세요 (예: normal, left_lean): ").strip()
    
    if not label:
        print("라벨이 필요합니다. 스크립트를 종료합니다.")
        sys.exit()
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{label}_{timestamp}.csv"
    
    # --- CSV 헤더 및 ID 정의 (MPU1, MPU4 제외) ---
    IMU_LABELS = ['under_body2', 'under_body3']
    IMU_IDS = [2, 3] # 수집할 MPU ID (순서대로)
    
    # 총 2개 IMU (6 + 6 = 12열)
    csv_header = create_csv_header(IMU_LABELS)
    
    print("-" * 30)
    print(f"라벨: {label}")
    print(f"파일명: {csv_filename}")
    print(f"총 열 개수: {len(csv_header)}")
    print(f"수집 대상 레이블: {IMU_LABELS}")
    print(f"수집 대상 ID: {IMU_IDS}")
    print("-" * 30)
    
    # --- 로깅 시작 (스레드 없음) ---
    start_logging(SERIAL_PORT, BAUD_RATE, csv_filename, csv_header, IMU_LABELS, IMU_IDS)