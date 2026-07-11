# HOW TO RUN

이 문서는 Jetson Orin Nano Super에서 CSI 카메라 기반 YOLO11n TensorRT + ByteTrack 실시간 사람 추적 코드를 실행하는 절차를 설명한다.

## 1. 실행 전 조건

검증된 기준 환경:

```text
Device       Jetson Orin Nano Super
L4T          R36.4.7
Python       3.10
CUDA         12.6
TensorRT     10.3
Ultralytics  8.4.46
OpenCV       Jetson system OpenCV 4.8.0
Camera       CSI camera, sensor-id 0
```

필수 파일:

```text
experiments/live_camera/hut_live_camera_track.py
models/engines/yolo11n_trt_fp16_imgsz640_static.engine
```

주의:

- TensorRT 엔진은 장치 및 TensorRT 버전에 종속될 수 있다.
- 일반 PC나 다른 Jetson 환경에서는 엔진 재생성이 필요할 수 있다.
- CSI 입력은 GStreamer 지원 OpenCV가 필요하다.
- `pip install opencv-python`으로 시스템 OpenCV를 덮어쓰지 않는다.

## 2. 저장소 받기

```bash
cd ~
git clone https://github.com/moongun515/jetson_state_replay.git
cd ~/jetson_state_replay
```

이미 clone한 경우:

```bash
cd ~/jetson_state_replay
git pull
```

## 3. 파일 확인

```bash
ls -lh experiments/live_camera/hut_live_camera_track.py
ls -lh models/engines/yolo11n_trt_fp16_imgsz640_static.engine
```

두 파일이 모두 출력되어야 한다.

## 4. Python 환경 활성화

검증에 사용한 가상환경:

```bash
source ~/yolo_test/yolo_env/bin/activate
```

확인:

```bash
which python
python --version
python - <<'PY'
import torch
from ultralytics import YOLO

print("CUDA available :", torch.cuda.is_available())
print("Torch version  :", torch.__version__)
print("Ultralytics OK")
PY
```

`CUDA available : True`가 출력되어야 한다.

가상환경 경로가 다르면 해당 환경의 `bin/activate`를 사용한다.

## 5. CSI 카메라 확인

### 5.1 Argus 카메라 단독 확인

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=60 !   'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' !   nvvidconv ! fakesink
```

정상 종료되면 카메라와 Argus 경로가 작동하는 것이다.

### 5.2 Python capture-only 확인

```bash
python experiments/live_camera/hut_live_camera_track.py   --source csi   --sensor-id 0   --cap-width 1280   --cap-height 720   --cap-fps 30   --capture-only   --duration 10   --display
```

`q` 또는 `ESC`로 종료할 수 있다.

실행 시작 시 다음과 유사하게 출력되어야 한다.

```text
OpenCV path     : /usr/lib/python3.10/dist-packages/cv2/__init__.py
OpenCV version  : 4.8.0
GStreamer       : True
```

## 6. 최종 실시간 데모

```bash
cd ~/jetson_state_replay
source ~/yolo_test/yolo_env/bin/activate

python experiments/live_camera/hut_live_camera_track.py   --source csi   --sensor-id 0   --cap-width 1280   --cap-height 720   --cap-fps 30   --model models/engines/yolo11n_trt_fp16_imgsz640_static.engine   --imgsz 640   --conf 0.40   --warmup-frames 30   --start-delay 5   --duration 60   --display   --save-jsonl outputs/state_json/live_demo_state.jsonl   --save-metrics outputs/metrics/live/live_demo_metrics.csv
```

동작:

```text
30프레임 TensorRT warm-up
→ 5초 카운트다운
→ 60초 실시간 사람 탐지 및 추적
→ State JSONL과 Metrics CSV 저장
```

## 7. 진단 영상 저장

```bash
python experiments/live_camera/hut_live_camera_track.py   --source csi   --sensor-id 0   --cap-width 1280   --cap-height 720   --cap-fps 30   --model models/engines/yolo11n_trt_fp16_imgsz640_static.engine   --imgsz 640   --conf 0.40   --warmup-frames 30   --start-delay 5   --duration 20   --save-video outputs/replay/live_debug_overlay.mp4   --save-raw-video outputs/replay/live_debug_raw.mp4   --save-jsonl outputs/state_json/live_debug_state.jsonl   --save-metrics outputs/metrics/live/live_debug_metrics.csv
```

영상 인코딩은 추가 자원을 사용하므로 최종 성능 측정에서는 영상 저장을 끈다.

## 8. 25W 최종 성능 측정

### 8.1 전력 모드 확인

```bash
sudo nvpmodel -q --verbose
```

검증 장치에서는 mode 1이 25W였다. 실제 출력에서 확인한 뒤 설정한다.

```bash
sudo nvpmodel -m 1
sudo jetson_clocks
sudo nvpmodel -q
```

### 8.2 출력 폴더 생성

```bash
cd ~/jetson_state_replay
mkdir -p outputs/metrics/live/final_25w
mkdir -p outputs/state_json/final_25w
```

### 8.3 tegrastats 시작

```bash
sudo tegrastats   --interval 1000   --logfile outputs/metrics/live/final_25w/tegrastats_csi_720p_640_conf040.log &
TEGRA_PID=$!
```

### 8.4 3분 benchmark 실행

```bash
python experiments/live_camera/hut_live_camera_track.py   --source csi   --sensor-id 0   --cap-width 1280   --cap-height 720   --cap-fps 30   --model models/engines/yolo11n_trt_fp16_imgsz640_static.engine   --imgsz 640   --conf 0.40   --warmup-frames 30   --start-delay 5   --duration 180   --save-jsonl outputs/state_json/final_25w/live_csi_720p_640_conf040_state.jsonl   --save-metrics outputs/metrics/live/final_25w/live_csi_720p_640_conf040_metrics.csv
```

### 8.5 tegrastats 종료

```bash
sudo kill "$TEGRA_PID"
```

### 8.6 결과 확인

```bash
tail -n 5   outputs/metrics/live/final_25w/live_csi_720p_640_conf040_metrics.csv

tail -n 10   outputs/metrics/live/final_25w/tegrastats_csi_720p_640_conf040.log
```

검증 결과:

```text
Processed frames  5,393
Measurement time  약 180초
Average FPS       29.958
RAM               약 2.96 GB
Swap              0 MB
Temperature       약 49°C
VDD_IN avg        약 5.94 W
```

## 9. 완전 오프라인 실행 확인

```bash
nmcli radio wifi off
```

유선 연결도 끄려면:

```bash
nmcli networking off
```

그다음 6번 또는 8번 명령을 실행한다.

다시 네트워크 활성화:

```bash
nmcli networking on
nmcli radio wifi on
```

## 10. 출력 결과 확인

```bash
tail -n 1 outputs/state_json/live_demo_state.jsonl | python -m json.tool
tail -n 5 outputs/metrics/live/live_demo_metrics.csv
```

## 11. 640과 960 비교

```text
imgsz 640 : 29.958 FPS, 최종 실시간 설정
imgsz 960 : 약 18.99 FPS, 정확도 우선 비교 설정
```

960은 세밀한 입력이 필요한 비교 실험용이며, 최종 제출 설정은 640이다.

## 12. 자주 발생하는 문제

### 12.1 `No cameras available`

```bash
sudo systemctl restart nvargus-daemon
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 ! fakesink
```

계속 실패하면 CSI 리본 방향, 커넥터 잠금, sensor-id 0/1, Jetson-IO 카메라 설정과 재부팅 여부를 확인한다.

### 12.2 `GStreamer: False`

OpenCV 경로가 다음인지 확인한다.

```text
/usr/lib/python3.10/dist-packages/cv2/__init__.py
```

일반 `opencv-python`은 Jetson CSI용 GStreamer를 지원하지 않을 수 있다.

### 12.3 모델 파일을 찾지 못함

```bash
find models -type f \( -name "*.engine" -o -name "*.pt" \) | sort
```

최종 엔진:

```text
models/engines/yolo11n_trt_fp16_imgsz640_static.engine
```

### 12.4 TensorRT 엔진 호환 오류

실행 장치의 GPU, JetPack 또는 TensorRT 버전이 다르면 엔진을 해당 장치에서 다시 생성해야 한다.

### 12.5 첫 프레임이 느림

TensorRT 초기화 비용이므로 `--warmup-frames 30`을 사용한다.

### 12.6 사람 중복 탐지

카메라 방향, 촬영 거리, 신체 잘림과 confidence를 확인한다.

권장:

```text
가로 방향
전신 또는 신체 대부분 확보
confidence 0.40
```

### 12.7 화면 밖으로 나갔다 돌아오면 ID가 바뀜

ByteTrack은 장시간 보이지 않은 트랙을 종료하므로 동일 인물이 재입장해도 새 ID가 부여될 수 있다.

## 13. 종료

실시간 화면 실행 중 `q` 또는 `ESC`로 종료한다.

카메라가 잠긴 경우:

```bash
sudo systemctl restart nvargus-daemon
```
