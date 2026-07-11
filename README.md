# Jetson State Replay

Jetson Orin Nano Super에서 CSI 카메라 영상을 직접 입력받아 사람을 탐지·추적하고, 프레임별 상태를 `State JSONL`로 저장하는 온디바이스 영상 상태 추출 프로젝트이다.

## 1. 최종 파이프라인

```text
CSI Camera
→ Jetson Argus / ISP
→ GStreamer
→ OpenCV
→ YOLO11n TensorRT FP16
→ ByteTrack
→ State JSONL + Metrics CSV
```

외부 서버나 클라우드 API를 사용하지 않는다. 모델 추론, 객체 추적, 상태 생성과 결과 저장이 모두 Jetson 내부에서 수행된다.

## 2. 최종 실시간 구성

| 항목 | 최종 설정 |
|---|---|
| Device | Jetson Orin Nano Super |
| Power mode | 25W |
| Camera | CSI |
| Camera input | 1280×720, 30 FPS |
| Detector | YOLO11n |
| Runtime | TensorRT FP16 |
| Model input | 640 |
| Detection class | person only |
| Confidence | 0.40 |
| Tracker | ByteTrack |
| State output | JSONL |
| Performance output | CSV + tegrastats |

최종 TensorRT 엔진:

```text
models/engines/yolo11n_trt_fp16_imgsz640_static.engine
```

## 3. 최종 성능 결과

25W 모드에서 CSI 카메라를 사용하여 3분 동안 실행한 결과이다. 화면 출력과 영상 저장은 비활성화하고 State JSONL과 Metrics CSV만 저장하였다.

| 항목 | 640 입력 |
|---|---:|
| Measurement time | 약 180초 |
| Processed frames | 5,393 |
| End-to-End FPS | 29.958 |
| Camera target FPS | 30 |
| CSI capture | 약 1~3 ms |
| YOLO + ByteTrack | 약 30~31 ms |
| RAM | 약 2.96 / 7.62 GB |
| Swap | 0 MB |
| Temperature | 약 49°C |
| VDD_IN 누적 평균 | 약 5.94 W |

30 FPS 카메라 입력을 거의 모두 처리했으며, 3분 동안 뚜렷한 성능 저하나 열 스로틀링은 관찰되지 않았다.

### 입력 크기 비교

| Model input | End-to-End FPS | 판단 |
|---:|---:|---|
| 640 | 29.958 | 최종 실시간 설정 |
| 960 | 약 18.99 | 정확도 우선 비교 설정 |

960 입력에서는 추론·추적 시간이 약 50~54 ms로 증가하여 실시간 30 FPS를 유지하지 못했다. 따라서 정확도와 실시간성의 균형점으로 640을 선택하였다.

## 4. 주요 파일

```text
jetson_state_replay/
├── README.md
├── HOW_TO_RUN.md
├── experiments/
│   └── live_camera/
│       └── hut_live_camera_track.py
├── models/
│   └── engines/
│       └── yolo11n_trt_fp16_imgsz640_static.engine
├── outputs/
│   ├── state_json/
│   ├── replay/
│   └── metrics/
└── src/
```

`hut_live_camera_track.py`라는 파일명은 개발 과정에서 사용한 이름이며, 현재 코드는 HUT 전용 로직이 아니라 일반적인 CSI 실시간 사람 탐지·추적 모듈이다.

## 5. 빠른 실행

동일한 Jetson 및 Python 환경에서:

```bash
cd ~/jetson_state_replay
source ~/yolo_test/yolo_env/bin/activate

python experiments/live_camera/hut_live_camera_track.py   --source csi   --sensor-id 0   --cap-width 1280   --cap-height 720   --cap-fps 30   --model models/engines/yolo11n_trt_fp16_imgsz640_static.engine   --imgsz 640   --conf 0.40   --warmup-frames 30   --start-delay 5   --duration 60   --display
```

상세 설치, 카메라 확인, 벤치마크 및 오류 해결 방법은 [HOW_TO_RUN.md](HOW_TO_RUN.md)를 참고한다.

## 6. 출력 파일

기본 출력:

```text
outputs/state_json/live_csi_test_state.jsonl
outputs/metrics/live/live_csi_test_metrics.csv
```

권장 최종 벤치마크 출력:

```text
outputs/state_json/final_25w/live_csi_720p_640_conf040_state.jsonl
outputs/metrics/live/final_25w/live_csi_720p_640_conf040_metrics.csv
outputs/metrics/live/final_25w/tegrastats_csi_720p_640_conf040.log
```

### State JSONL 주요 필드

```json
{
  "frame": 100,
  "timestamp": 3.34,
  "scene_state": "OCCUPIED",
  "object_count": 1,
  "objects": [
    {
      "id": 1,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "bbox_xyxy": [320.1, 95.4, 713.8, 715.2],
      "center": [516.95, 405.3],
      "velocity_px_s": [2.1, -1.3],
      "speed_px_s": 2.47,
      "acceleration_px_s2": [0.4, -0.2]
    }
  ]
}
```

### Metrics CSV 주요 필드

```text
frame
timestamp_sec
capture_ms
infer_track_ms
state_ms
display_ms
total_loop_ms
person_count
average_fps
recent_fps
```

## 7. 화면 표시 의미

진단 overlay에서:

```text
TRACK id:n = ByteTrack ID가 부여된 추적 객체
CAND id:-1 = 탐지는 되었지만 아직 추적 ID가 확정되지 않은 후보
```

사람이 화면 밖으로 완전히 나갔다가 다시 들어오면 ByteTrack이 새 ID를 부여할 수 있다. 이는 동일 인물 재식별이 아니라 프레임 간 객체 추적을 수행하는 ByteTrack의 정상적인 동작이다.

## 8. 카메라 방향

모델 정확도를 위해 영상 속 사람이 정상적으로 세워진 방향으로 입력되어야 한다. 카메라가 90도 회전된 상태에서는 신체 일부나 배경 물체를 사람으로 중복 탐지하는 현상이 증가했다.

권장 조건:

```text
가로 방향
전신 또는 신체 대부분이 보이는 거리
화면 가장자리에 충분한 여백
confidence 0.40
```

## 9. 환경 및 호환성

검증 환경:

```text
Jetson Orin Nano Super
JetPack / L4T R36.4.7
Python 3.10
CUDA 12.6
TensorRT 10.3
Ultralytics 8.4.46
Jetson system OpenCV 4.8.0 with GStreamer
```

TensorRT 엔진은 생성한 장치, JetPack, TensorRT 버전과의 호환성이 중요하다. 다른 장치나 다른 TensorRT 버전에서는 `.engine` 파일을 다시 생성해야 할 수 있다.

또한 CSI 입력을 위해 `pip`의 일반 `opencv-python`이 아니라 GStreamer가 활성화된 Jetson 시스템 OpenCV를 사용한다. 현재 live 스크립트는 다음 경로의 OpenCV를 우선 사용하도록 구성되어 있다.

```text
/usr/lib/python3.10/dist-packages/cv2
```

## 10. 기존 MOT17 실험

본 프로젝트는 MOT17 이미지 시퀀스를 이용한 YOLO11 + ByteTrack + State Replay 성능 실험에서 시작하였다. 이후 저장 이미지 입력의 JPG 디코딩 병목을 줄이고 실제 온디바이스 동작을 확인하기 위해 CSI 카메라 실시간 입력으로 확장하였다.

최종 결론:

> CSI 카메라와 Jetson ISP를 사용한 온디바이스 파이프라인에서 YOLO11n TensorRT FP16, 입력 크기 640, ByteTrack 조합은 25W 모드에서도 평균 29.958 FPS를 달성하였다. 입력 크기를 960으로 높이면 약 19 FPS로 감소하여, 정확도와 실시간성의 최종 균형점은 640으로 판단하였다.
