# 1️⃣ 현재 디렉토리를 /app으로 마운트, 이미지 이름은 gonghack-ai:latest
# 2️⃣ 시리얼 장치 2개 매핑 (STM 입력, ESP 출력)
# 3️⃣ 모든 포트 5000~6000 개방
docker run -it --rm \
    --device /dev/ttyACM0:/dev/ttyACM0 \
    --device /dev/ttyUSB0:/dev/ttyUSB0 \
    --privileged \
    -p 5000-6000:5000-6000 \
    -v $(pwd):/app \
    gonghack-ai:latest
