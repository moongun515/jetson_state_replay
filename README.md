# Jetson State Replay Project

## 1. Project Goal

이 프로젝트의 목표는 영상에서 객체의 상태를 추출하여 `State JSON`으로 구조화하고, 이 JSON을 기반으로 원본 영상 또는 추상화된 2D 장면을 재현하는 것이다.

최종적으로는 Jetson Orin Nano Super 환경에서 객체 탐지, 추적, State JSON 생성, 2D Replay 생성 과정을 실행하고, 처리 속도와 리소스 사용량을 측정하는 것을 목표로 한다.

```text
Video / Image Sequence
→ Object Detection
→ Object Tracking
→ State JSON
→ 2D Replay
→ Performance Measurement
```

---

## 2. Current Dataset

현재 MVP 실험에는 MOT17 데이터셋의 `MOT17-02-DPM` 시퀀스를 사용하였다.

```text
data/
└─ MOT17-02-DPM/
   ├─ img1/
   │  └─ 000001.jpg ~ 000600.jpg
   ├─ gt/
   │  └─ gt.txt
   └─ seqinfo.ini
```

Sequence 정보:

```text
name      = MOT17-02-DPM
frameRate = 30
seqLength = 600
imWidth   = 1920
imHeight  = 1080
imExt     = .jpg
```

---

## 3. Basic Folder Structure

```text
jetson_state_replay/
├─ data/
│  └─ MOT17-02-DPM/
│     ├─ img1/
│     ├─ gt/
│     │  └─ gt.txt
│     └─ seqinfo.ini
│
├─ outputs/
│  ├─ state_json/
│  ├─ replay/
│  └─ metrics/
│
├─ src/
│  ├─ mot17_gt_to_state_json.py
│  ├─ replay_2d_from_state.py
│  ├─ yolo_bytetrack_to_state_json.py
│  └─ make_gt_pred_summary.py
│
├─ experiments/
└─ README.md
```

---

## 4. Pipeline 1: GT Label Based State Replay

첫 번째 파이프라인은 MOT17의 정답 라벨인 `gt.txt`를 사용한다.

```text
MOT17 gt.txt
→ mot17_gt_to_state_json.py
→ mot17_02_gt_state.json
→ replay_2d_from_state.py
→ mot17_02_gt_overlay.mp4
→ mot17_02_gt_abstract.mp4
```

GT 기반 Replay는 정답 라벨이 포함하고 있는 pedestrian 객체를 State JSON으로 변환하고, 이를 원본 영상 위에 overlay하거나 검은 배경 위에 abstract replay로 재현한다.

생성 파일:

```text
outputs/state_json/mot17_02_gt_state.json
outputs/replay/mot17_02_gt_overlay.mp4
outputs/replay/mot17_02_gt_abstract.mp4
outputs/metrics/mot17_02_gt_replay_metrics.json
```

---

## 5. Pipeline 2: YOLO + ByteTrack Based Automatic Replay

두 번째 파이프라인은 정답 라벨을 사용하지 않고, YOLO11과 ByteTrack을 이용해 자동으로 객체 상태를 추출한다.

```text
MOT17 img1 frames
→ YOLO11 object detection
→ ByteTrack object tracking
→ yolo_bytetrack_to_state_json.py
→ mot17_02_pred_yolo_bytetrack_state.json
→ replay_2d_from_state.py
→ mot17_02_pred_yolo_bytetrack_overlay.mp4
→ mot17_02_pred_yolo_bytetrack_abstract.mp4
```

생성 파일:

```text
outputs/state_json/mot17_02_pred_yolo_bytetrack_state.json
outputs/replay/mot17_02_pred_yolo_bytetrack_overlay.mp4
outputs/replay/mot17_02_pred_yolo_bytetrack_abstract.mp4
outputs/metrics/mot17_02_pred_yolo_bytetrack_metrics.json
outputs/metrics/mot17_02_pred_yolo_bytetrack_replay_metrics.json
```

---

## 6. State JSON Format

State JSON은 frame 단위로 객체 상태를 저장한다.

예시:

```json
{
  "frame": 1,
  "timestamp": 0.0,
  "object_count": 1,
  "objects": [
    {
      "id": 2,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.87,
      "bbox_xywh": [1338.0, 418.0, 167.0, 379.0],
      "bbox_xyxy": [1338.0, 418.0, 1505.0, 797.0],
      "center": [1421.5, 607.5],
      "velocity": [0.0, 0.0],
      "raw_velocity": [0.0, 0.0],
      "source": "yolo_bytetrack"
    }
  ]
}
```

주요 필드:

| Field | Meaning |
|---|---|
| `frame` | 프레임 번호 |
| `timestamp` | 시간 |
| `id` | 객체 추적 ID |
| `class_name` | 객체 클래스 이름 |
| `confidence` | YOLO 예측 신뢰도 |
| `bbox_xywh` | x, y, width, height |
| `bbox_xyxy` | x1, y1, x2, y2 |
| `center` | 객체 중심 좌표 |
| `velocity` | smoothing된 중심 이동 벡터 |
| `raw_velocity` | 원본 중심 이동 벡터 |

---

## 7. 2D Replay Modes

현재 Replay는 두 가지 방식으로 생성된다.

### 7.1 Overlay Replay

원본 이미지 위에 bbox, id, center, velocity arrow, trajectory를 표시한다.

목적:

- 라벨 또는 예측 결과가 실제 영상에 잘 맞는지 확인
- GT와 YOLO+ByteTrack 결과 비교

### 7.2 Abstract Replay

검은 배경 위에 객체 상태만 표시한다.

목적:

- 원본 영상 없이 State JSON만으로 가상 장면 재현
- 객체 이동 경로와 추적 ID 확인

---

## 8. Initial Quantitative Result

초기 MOT17-02-DPM 600프레임 기준 결과는 다음과 같다.

| Item | GT | YOLO+ByteTrack |
|---|---:|---:|
| Frame count | 600 | 600 |
| Total objects | 18,581 | 5,052 |
| Avg objects/frame | 약 30.97 | 약 8.42 |
| YOLO+ByteTrack extraction time | - | 약 59.73 sec |
| YOLO+ByteTrack extraction FPS | - | 약 10.05 FPS |

GT는 정답 라벨 기반이므로 원거리, 작은 사람, 부분적으로 가려진 사람까지 매우 촘촘하게 포함한다.

YOLO+ByteTrack은 자동 추출 기반이므로 가까운 객체, 선명한 객체, 전경에 있는 객체는 안정적으로 탐지하지만, 멀리 있는 사람, 겹친 사람, 가려진 사람은 누락되거나 하나로 병합되는 경향이 있다.

---

## 9. Key Observations

- GT Replay는 매우 세밀하다.
- 작은 사람, 멀리 있는 사람, 일부 가려진 사람도 라벨링되어 있다.
- YOLO+ByteTrack Replay는 실제 모델이 인지 가능한 객체 중심이다.
- 가까운 사람은 잘 잡는다.
- 멀리 있는 사람은 confidence가 낮아 누락된다.
- 객체가 겹치면 앞에 있는 사람 중심으로 잡히거나 하나로 병합된다.
- 자전거 탑승자는 GT와 YOLO 기준이 다르게 나타날 수 있다.
- MOT17 GT는 pedestrian tracking 대상 기준이다.
- YOLO는 이미지에서 사람 형태가 보이면 `person`으로 탐지할 수 있다.
- ByteTrack은 YOLO가 검출한 bbox를 프레임 간 연결하는 역할이다.
- YOLO가 객체를 검출하지 못하면 ByteTrack도 해당 객체를 추적할 수 없다.

---

## 10. Basic Run Commands

### 10.1 GT State JSON 생성

```bash
python src/mot17_gt_to_state_json.py
```

### 10.2 YOLO + ByteTrack State JSON 생성

```bash
python src/yolo_bytetrack_to_state_json.py
```

### 10.3 2D Replay 생성

`replay_2d_from_state.py`의 `STATE_PATH`와 출력 파일명을 GT 또는 Pred에 맞게 설정한 뒤 실행한다.

```bash
python src/replay_2d_from_state.py
```

### 10.4 GT vs Pred Summary 생성

```bash
python src/make_gt_pred_summary.py
```

출력:

```text
outputs/metrics/mot17_02_gt_vs_pred_summary.json
outputs/metrics/mot17_02_gt_vs_pred_summary.csv
outputs/metrics/mot17_02_gt_vs_pred_summary.md
```

---

## 11. Current MVP Status

### MVP-0: GT 기반 State Replay

완료.

```text
MOT17 gt.txt
→ State JSON
→ Overlay Replay
→ Abstract Replay
```

### MVP-1: YOLO + ByteTrack 자동 추출 Replay

완료.

```text
MOT17 frames
→ YOLO11
→ ByteTrack
→ Pred State JSON
→ Overlay Replay
→ Abstract Replay
```

현재까지의 핵심 성과는 정답 라벨 기반 Replay와 AI 자동 추출 기반 Replay를 모두 구현하고, 두 결과의 차이를 시각적·수치적으로 비교할 수 있게 만든 것이다.

---

## 12. Jetson Orin Nano Super GPU Environment

Jetson Orin Nano Super에서 YOLO11 + ByteTrack 파이프라인을 GPU 기반으로 실행하기 위해 NVIDIA 호환 환경을 정리하였다.

현재 정상 동작이 확인된 환경은 다음과 같다.

| Item | Version / Status |
|---|---|
| Device | NVIDIA Jetson Orin Nano Super Developer Kit |
| CUDA | 12.6 |
| cuDNN | 9.3 |
| TensorRT | 10.3.0 |
| PyTorch | 2.8.0 |
| Torchvision | 0.23.0 |
| NumPy | 1.26.1 |
| OpenCV | 4.11.0 |
| Ultralytics | 8.4.46 |
| GPU availability | `torch.cuda.is_available() == True` |
| GPU device | `cuda:0` |

Jetson 환경은 일반 PC처럼 라이브러리를 무조건 최신 버전으로 업데이트하면 안 된다. JetPack, CUDA, cuDNN, TensorRT, PyTorch 간 호환성을 유지해야 한다.

특히 아래 명령은 목적 없이 실행하지 않는다.

```bash
pip install -U torch
pip install -U torchvision
pip install -U numpy
pip install -U opencv-python
pip install -U ultralytics
sudo apt upgrade
sudo apt full-upgrade
```

---

## 13. NVIDIA Optimization Process

초기에는 PyTorch CUDA 환경이 정상적으로 연결되지 않아 CPU fallback 상태로 실행되었다. 이후 Jetson 전용 PyTorch CUDA 환경을 복구하고, 클럭 고정과 TensorRT FP16 최적화를 단계적으로 적용하였다.

### 13.1 Optimization Summary

| Stage | Model | Runtime | Power / Clock Mode | Processing FPS |
|---|---|---|---|---:|
| Initial fallback | YOLO11n | CPU fallback | GPU disabled | 2.087 |
| CUDA recovery | YOLO11s `.pt` | PyTorch CUDA | 25W, dynamic clock | 10.149 |
| Clock optimization | YOLO11s `.pt` | PyTorch CUDA | 25W, `jetson_clocks` | 약 14.291 |
| Lightweight baseline | YOLO11n `.pt` | PyTorch CUDA | 25W, `jetson_clocks` | 14.604 |
| TensorRT FP16 | YOLO11n `.engine` | TensorRT FP16 | 25W, `jetson_clocks` | 약 23.788 |
| Final optimized | YOLO11n `.engine` | TensorRT FP16 | `MAXN_SUPER`, `jetson_clocks` | 약 29.060 |

초기 CPU fallback 상태와 최종 MAXN_SUPER TensorRT FP16 상태를 비교하면:

```text
2.087 FPS
→ 29.060 FPS
```

약 13.9배의 처리 속도 향상을 확인하였다.

---

## 14. TensorRT FP16 Conversion

TensorRT FP16 엔진은 다음 흐름으로 생성하였다.

```text
yolo11n.pt
→ yolo11n.onnx
→ yolo11n.engine
```

각 파일의 역할은 다음과 같다.

| File | Meaning |
|---|---|
| `.pt` | PyTorch 원본 모델 |
| `.onnx` | 프레임워크 간 변환을 위한 공통 중간 형식 |
| `.engine` | Jetson GPU에 맞게 최적화된 TensorRT 실행 엔진 |

변환 명령:

```bash
yolo export \
  model=yolo11n.pt \
  format=engine \
  imgsz=640 \
  half=True \
  device=0
```

Jetson에서 TensorRT 엔진을 생성하는 과정은 수 분 정도 걸릴 수 있다. YOLO11n FP16 엔진 생성에는 약 6~7분이 소요되었다.

`onnxruntime-gpu` 자동 설치 경고가 발생할 수 있지만, ONNX 변환과 TensorRT 엔진 생성이 성공하면 현재 TensorRT 실행에는 문제가 없다.

---

## 15. Jetson Orin Nano Super Power Modes

현재 장비에서 확인한 주요 전력 모드는 다음과 같다.

| Mode ID | Mode | Purpose |
|---:|---|---|
| `0` | `15W` | 저전력·저발열 실험용, 아직 미측정 |
| `1` | `25W` | 안정적인 일반 운용 기준 |
| `2` | `MAXN_SUPER` | 최고 성능 한계 측정 |

현재 프로젝트에서는 `25W`와 `MAXN_SUPER`를 집중 비교하였다.

### 15.1 25W Mode

25W 모드는 안정적인 일반 운용 기준으로 사용하였다.

설정:

```bash
sudo /usr/sbin/nvpmodel -m 1
sudo /usr/bin/jetson_clocks
```

3회 반복 결과:

| Run | Elapsed Time | FPS | Tracked Objects |
|---|---:|---:|---:|
| 1 | 25.5224 sec | 23.509 | 5,368 |
| 2 | 25.1152 sec | 23.890 | 5,368 |
| 3 | 25.0380 sec | 23.964 | 5,368 |
| Average | **25.2252 sec** | **23.788** | **5,368** |

### 15.2 MAXN_SUPER Mode

MAXN_SUPER는 Jetson Orin Nano Super의 최고 성능 한계를 확인하기 위해 사용하였다.

설정:

```bash
sudo /usr/sbin/nvpmodel -m 2
sudo /usr/bin/jetson_clocks
```

3회 반복 결과:

| Run | Elapsed Time | FPS | Tracked Objects |
|---|---:|---:|---:|
| 1 | 20.7490 sec | 28.917 | 5,368 |
| 2 | 20.6306 sec | 29.083 | 5,368 |
| 3 | 20.5612 sec | 29.181 | 5,368 |
| Average | **20.6469 sec** | **29.060** | **5,368** |

### 15.3 Power Mode Comparison

| Item | 25W | MAXN_SUPER | Difference |
|---|---:|---:|---:|
| Average FPS | 23.788 | **29.060** | 약 **22.2% 증가** |
| Average elapsed time | 25.2252 sec | **20.6469 sec** | 약 **18.1% 감소** |
| Tracked objects | 5,368 | 5,368 | 동일 |

MAXN_SUPER 적용 시 tracked object 수는 유지하면서 처리 속도만 향상되었다.

---

## 16. Resource Usage Observation

`tegrastats`를 사용하여 RAM, CPU, GPU, 온도, 전력을 확인하였다.

실행 예시:

```bash
tegrastats --interval 1000 \
  --logfile outputs/metrics/tegrastats_test.log &
```

종료:

```bash
pkill tegrastats
```

후반부 대표 구간을 기준으로 비교하면 다음과 같다.

| Item | 25W | MAXN_SUPER |
|---|---:|---:|
| CPU temperature | 약 51.6°C | 약 52.7°C |
| GPU temperature | 약 51.5°C | 약 52.6°C |
| Board input power `VDD_IN` | 약 7.21W | 약 7.84W |
| RAM usage | 약 4.33GB / 7.62GB | 약 4.25GB / 7.62GB |
| Swap usage | 약 66MB | 약 66MB |

MAXN_SUPER 적용 시 온도 상승은 약 1.2°C 수준이었고, 반복 실행 중 thermal throttling이나 메모리 부족 현상은 관찰되지 않았다.

단, 위 수치는 전체 로그 평균이 아니라 각 로그 후반부 대표 구간을 기준으로 한 관찰값이다.

---

## 17. Current Bottleneck Analysis

TensorRT FP16 적용 이후 GPU 추론 자체는 충분히 빨라졌다.

다만 `tegrastats` 로그에서는 특정 CPU 코어 사용률이 반복적으로 90% 이상까지 상승하였다. 따라서 현재 파이프라인의 추가 병목은 다음 구간에 있을 가능성이 있다.

```text
Image read
→ Preprocess
→ TensorRT inference
→ Postprocess
→ ByteTrack
→ State JSON generation
```

향후 추가 성능 개선에서는 GPU 클럭을 더 올리는 것보다 CPU 기반 전처리, 후처리, ByteTrack, JSON 생성 비용을 분리 측정하는 것이 중요하다.

---

## 18. Detection Policy Caveat

현재 YOLO + ByteTrack State JSON은 `person` 클래스만 저장한다.

TensorRT FP16 최종 결과:

```text
Total tracked objects: 5,368
Stored classes: person only
```

다만 영상 확인 결과, 일부 자전거 탑승자의 사람 영역도 `person`으로 탐지되었다.

```text
MOT17 GT
→ pedestrian tracking policy

YOLO
→ visually detected person objects
```

따라서 GT 객체 수와 YOLO 객체 수의 차이는 단순 탐지 실패뿐 아니라 라벨링 정책 차이의 영향도 포함한다.

객체 수 증가만으로 정확도 향상을 단정하지 않고, 원거리 보행자 누락, 겹침, 가림, ID 유지, 자전거 탑승자 포함 여부를 함께 분석해야 한다.

---

## 19. Final Folder Structure

최종 모델과 결과 파일은 다음과 같이 정리하였다.

```text
jetson_state_replay/
├─ models/
│  ├─ final/
│  │  ├─ yolo11n.pt
│  │  ├─ yolo11n.onnx
│  │  ├─ yolo11n_25w.engine
│  │  └─ yolo11n_maxn_super.engine
│  └─ archive/
│     ├─ yolo11s.pt
│     └─ yolo11s.onnx
│
├─ outputs/
│  ├─ metrics/
│  │  ├─ final/
│  │  │  ├─ mot17_02_yolo11n_trt_fp16_25w_metrics.json
│  │  │  ├─ mot17_02_yolo11n_trt_fp16_maxn_super_metrics.json
│  │  │  └─ tegrastats/
│  │  │     ├─ 25w_run1.log
│  │  │     ├─ 25w_run2.log
│  │  │     ├─ 25w_run3.log
│  │  │     ├─ maxn_super_run1.log
│  │  │     ├─ maxn_super_run2.log
│  │  │     └─ maxn_super_run3.log
│  │  └─ archive/
│  │
│  ├─ state_json/
│  │  ├─ final/
│  │  │  ├─ mot17_02_yolo11n_trt_fp16_25w_state.json
│  │  │  └─ mot17_02_yolo11n_trt_fp16_maxn_super_state.json
│  │  └─ archive/
│  │
│  └─ replay/
│     └─ final/
│        ├─ mot17_02_gt_overlay.mp4
│        ├─ mot17_02_gt_abstract.mp4
│        ├─ mot17_02_yolo11n_trt_fp16_overlay.mp4
│        └─ mot17_02_yolo11n_trt_fp16_abstract.mp4
│
└─ src/
   ├─ mot17_gt_to_state_json.py
   ├─ replay_2d_from_state.py
   ├─ yolo_bytetrack_to_state_json.py
   └─ make_gt_pred_summary.py
```

---

## 20. Recommended Runtime Modes

| Purpose | Recommended Setting |
|---|---|
| Stable general operation | 25W + TensorRT FP16 |
| Maximum performance test | MAXN_SUPER + TensorRT FP16 |
| Low-power experiment | 15W mode, not measured yet |
| Long-running camera operation | Start with 25W, then compare MAXN_SUPER after thermal testing |

---

## 21. Important Jetson Cautions

- Jetson은 일반 PC처럼 무조건 최신 버전으로 업데이트하면 안 된다.
- JetPack, CUDA, cuDNN, TensorRT, PyTorch 버전 조합을 맞춰서 사용해야 한다.
- 현재 정상 환경은 최대한 유지하고 불필요한 설치는 피한다.
- 코드 안의 `import`는 자유롭게 사용 가능하다.
- 위험한 것은 설치 및 업데이트 명령이다.
- 기존 `yolo_env`는 기준 환경으로 보존한다.
- 새로운 실험은 새 가상환경에서 진행한다.
- 외부 프로젝트의 `requirements.txt`는 바로 설치하지 않고 먼저 내용을 확인한다.
- CSI 카메라 연결은 가능하면 전원을 끄고 작업한다.
- 추론 또는 영상 저장 중 전원을 바로 뽑지 않는다.
- 종료는 `sudo shutdown -h now`를 사용한다.
- 팬 정지, 화면 꺼짐, 재부팅 반복 시 코드보다 전원과 발열부터 확인한다.

---

## 22. Next Steps

1. 전체 `tegrastats` 로그를 파싱하여 평균 / 최대 전력, RAM, GPU 사용률, 온도를 자동 계산한다.
2. 10~30분 연속 실행으로 장시간 thermal stability를 확인한다.
3. CSI 카메라를 연결하고 실시간 입력을 테스트한다.
4. preprocess, TensorRT inference, postprocess, ByteTrack, State JSON 생성 시간을 분리 측정한다.
5. confidence threshold와 input resolution 조건을 비교한다.
6. `person`과 `bicycle`이 겹치는 경우 rider candidate 태그를 추가하는 방안을 검토한다.
7. 15W 모드를 저전력 기준으로 추가 측정한다.
