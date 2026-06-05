# Jetson State Replay Project

## 1. Project Goal

이 프로젝트의 목표는 영상에서 객체 상태를 추출하여 `State JSON`으로 구조화하고, 해당 JSON을 기반으로 2D 장면을 재현하는 것이다.

최종적으로는 Jetson Orin Nano Super에서 다음 파이프라인을 실시간에 가깝게 실행하고, 처리 속도와 자원 한계를 측정한다.

```text
Video / Image Sequence / CSI Camera
→ YOLO11 Object Detection
→ ByteTrack Object Tracking
→ State JSON
→ Optional 2D Replay / Preview
→ Performance Measurement
```

현재 우선순위는 Replay를 무겁게 붙이는 것이 아니라, **입력부터 State 생성까지의 실시간 추출 성능을 먼저 확보하는 것**이다.

---

## 2. Core Pipeline

### 2.1 GT Label Based Replay

```text
MOT17 gt.txt
→ mot17_gt_to_state_json.py
→ GT State JSON
→ replay_2d_from_state.py
→ Overlay Replay / Abstract Replay
```

### 2.2 YOLO + ByteTrack Automatic Extraction

```text
MOT17 img1 frames
→ YOLO11n TensorRT FP16
→ ByteTrack
→ bbox / center / velocity 계산
→ State JSON
→ Optional Replay
```

### 2.3 Current Optimization Direction

```text
Reader Thread
JPG read + decode
        ↓
Queue(maxsize=4)
        ↓
Main Thread
TensorRT YOLO11n + ByteTrack
        ↓
State JSON object build
```

기존 직렬 구조에서는 JPG 읽기와 추론이 순차적으로 실행되었다.  
현재는 제한된 Queue를 사용해 입력 디코딩과 추론을 겹쳐 처리한다.

`Queue(maxsize=4)`는 전체 영상을 RAM에 쌓는 preload 방식이 아니다.  
디코딩된 프레임을 최대 4장만 유지하므로 메모리 증가를 제한한다.

---

## 3. Current Dataset

### 3.1 MOT17-02-DPM

```text
Resolution : 1920 x 1080
FPS        : 30
Frames     : 600
Purpose    : 기본 정확도 검증 및 초기 성능 측정
```

### 3.2 MOT17-04-DPM

```text
Resolution : 1920 x 1080
FPS        : 30
Frames     : 1050
Purpose    : 객체 밀도가 더 높은 스트레스 테스트
```

### 3.3 MOT17-03-DPM

```text
Resolution : 1920 x 1080
FPS        : 30
Frames     : 1500
Purpose    : 더 강한 고밀도 스트레스 테스트
Note       : test split이므로 GT 정확도 비교보다 성능 한계 측정용
```

---

## 4. Folder Structure

```text
jetson_state_replay/
├─ data/
│  ├─ MOT17-02-DPM/
│  ├─ MOT17-04-DPM/
│  └─ MOT17-03-DPM/
│
├─ src/
│  ├─ mot17_gt_to_state_json.py
│  ├─ replay_2d_from_state.py
│  ├─ yolo_bytetrack_to_state_json.py
│  └─ make_gt_pred_summary.py
│
├─ experiments/
│  ├─ timing/
│  │  ├─ mot17_trt_stage_timing.py
│  │  └─ mot17_trt_preload_timing.py
│  └─ cpu_queue/
│     ├─ mot17_trt_async_reader.py
│     ├─ mot17_trt_async_reader_stress_v2.py
│     └─ mot17_trt_async_reader_sequence_stress.py
│
├─ outputs/
│  ├─ state_json/
│  ├─ replay/
│  ├─ metrics/
│  └─ logs/
│
└─ README.md
```

---

## 5. State JSON Format

State JSON은 frame 단위로 객체 상태를 저장한다.

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
| `id` | ByteTrack 객체 ID |
| `class_name` | 객체 클래스 |
| `confidence` | YOLO 신뢰도 |
| `bbox_xywh` | x, y, width, height |
| `bbox_xyxy` | x1, y1, x2, y2 |
| `center` | bbox 중심 좌표 |
| `velocity` | smoothing된 이동 벡터 |
| `raw_velocity` | 원본 중심 이동 벡터 |

---

## 6. Completed MVP

### MVP-0. GT Label Replay

완료.

```text
MOT17 gt.txt
→ State JSON
→ Overlay Replay
→ Abstract Replay
```

### MVP-1. YOLO + ByteTrack Automatic Replay

완료.

```text
MOT17 frames
→ YOLO11
→ ByteTrack
→ Pred State JSON
→ Overlay Replay
→ Abstract Replay
```

### MVP-2. Jetson TensorRT FP16 Extraction

완료.

```text
MOT17 JPG frames
→ YOLO11n TensorRT FP16
→ ByteTrack
→ State JSON
```

### MVP-3. CPU Input Pipeline Optimization

완료.

```text
Reader Thread
→ Queue(maxsize=4)
→ TensorRT YOLO11n + ByteTrack
→ State JSON
```

---

## 7. Baseline GT vs Prediction Result

MOT17-02-DPM 600프레임 기준 초기 비교:

| Item | GT | YOLO + ByteTrack |
|---|---:|---:|
| Frame count | 600 | 600 |
| Total objects | 18,581 | 5,052 |
| Avg objects/frame | 30.968 | 8.420 |
| Initial extraction FPS | - | 10.046 |

관찰:

- GT는 원거리, 작은 사람, 가려진 사람까지 촘촘하게 포함한다.
- YOLO + ByteTrack은 가까운 객체와 비교적 선명한 객체를 안정적으로 탐지한다.
- 원거리, 겹침, 가림이 심한 사람은 누락되거나 병합될 수 있다.
- MOT17 GT 정책과 일반 객체 탐지기의 person 판단 기준은 완전히 같지 않다.

---

## 8. Stage Timing Measurement

`mot17_trt_stage_timing.py`를 사용하여 프레임별 구간 시간을 측정하였다.

기록 항목:

```text
frame
raw_detection_count
target_detection_count
active_track_count
read_ms
inference_tracking_ms
state_build_ms
total_ms
instant_fps
```

MAXN_SUPER 직렬 입력 측정 예시:

| Stage | Approximate latency |
|---|---:|
| JPG read + decode | 약 12 ms |
| TensorRT inference + ByteTrack | 약 19~23 ms |
| State build | 약 1~1.5 ms |
| Total | 약 34 ms |

판단:

- State build는 객체 수 증가에 따라 늘어나지만 절대 시간은 비교적 작다.
- 주요 병목은 JPG 입력 처리와 TensorRT + ByteTrack 구간이다.
- 입력 처리와 추론을 겹쳐 실행할 가치가 있다.

---

## 9. Preload Diagnostic

전체 JPG 프레임을 디코딩하여 RAM에 올리는 preload 진단을 수행하였다.

```text
1920 x 1080 x 3 bytes x 600 frames
≈ 3.56 GiB
```

결과:

```text
Decoded frame RAM usage: 3559.6 MiB
CUDA initialization failure
out of memory
segmentation fault
```

판단:

- 전체 영상 preload는 Jetson 실시간 구조에 부적합하다.
- Jetson에서는 프레임을 쌓아두기보다 제한된 버퍼를 통해 흐르게 해야 한다.
- 최종 구조는 작은 Queue 기반 streaming 방식이 적절하다.

---

## 10. Queue(maxsize=4) Power Mode Benchmark

조건:

```text
MOT17-02-DPM
YOLO11n TensorRT FP16
ByteTrack
person tracking
JPG Reader Thread
Queue(maxsize=4)
State build
```

25W와 MAXN_SUPER를 각각 3회 반복 측정하였다.

| Mode | Runs | Pipeline FPS mean | Std | Min | Max | Decode ms | Infer+track ms | State ms | Consumer ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25W | 3 | 37.871 | 0.236 | 37.656 | 38.124 | 15.646 | 23.546 | 1.596 | 25.146 |
| MAXN_SUPER | 3 | 45.448 | 0.144 | 45.292 | 45.575 | 12.377 | 19.558 | 1.395 | 20.957 |

핵심 결과:

```text
25W       : 평균 37.871 FPS
MAXN_SUPER: 평균 45.448 FPS
```

30 FPS 기준을 두 전력 모드에서 모두 넘겼다.

주의:

- 직렬 입력 결과와 Queue 결과는 파일 캐시, 온도, 실행 순서의 영향을 받을 수 있다.
- 따라서 개선율을 단정하기보다 Queue 구조의 반복 평균값을 기준 성능으로 사용한다.

---

## 11. Object Density Stress Test

Queue 구조를 유지한 채 시퀀스와 추적 클래스 범위를 바꾸어 스트레스 테스트를 수행하였다.

### 11.1 MAXN_SUPER Results

| Sequence | Class mode | Frames | Total objects | Avg objects/frame | Pipeline FPS |
|---|---|---:|---:|---:|---:|
| MOT17-02-DPM | person | 600 | 5,379 | 8.97 | 47.198 |
| MOT17-02-DPM | all | 600 | 6,503 | 10.84 | 44.185 |
| MOT17-04-DPM | person | 1,050 | 16,849 | 16.05 | 43.243 |
| MOT17-04-DPM | all | 1,050 | 18,515 | 17.63 | 41.987 |
| MOT17-03-DPM | person | 1,500 | 33,509 | 22.34 | 38.912 |
| MOT17-03-DPM | all | 1,500 | 38,916 | 25.94 | 37.750 |

관찰:

- 객체 수가 증가할수록 State build 시간이 늘어난다.
- 객체 밀도가 높은 MOT17-03-DPM에서도 MAXN_SUPER는 30 FPS 이상을 유지한다.
- `MOT17-03-DPM + all classes` 조건에서도 약 37.750 FPS를 기록하였다.

### 11.2 25W Worst-case Result

조건:

```text
MOT17-03-DPM
all classes
Queue(maxsize=4)
25W
```

| Item | Result |
|---|---:|
| Frames | 1,500 |
| Total objects | 38,916 |
| Avg objects/frame | 25.94 |
| Pipeline FPS | 30.795 |
| Avg consumer latency | 31.951 ms |
| P95 consumer latency | 35.158 ms |
| Avg queue wait | 0.027 ms |

판단:

- 25W에서도 평균 기준으로 30 FPS를 넘겼다.
- 다만 일부 프레임은 30 FPS 아래로 내려갔다.
- 혼잡 장면에서 안정적으로 30 FPS 이상을 유지하려면 MAXN_SUPER가 더 적절하다.

---

## 12. Current Interpretation

현재 상태 추출 파이프라인은 다음 수준까지 확인하였다.

```text
25W:
일반 장면에서 실시간 추출 가능
고밀도 다중 객체 조건에서도 평균 약 30 FPS 가능

MAXN_SUPER:
일반 장면에서 약 45 FPS
고밀도 다중 객체 조건에서도 약 38 FPS
추가 Preview 기능을 붙여볼 여유 존재
```

권장 사용 방향:

| Scenario | Recommended mode |
|---|---|
| 일반 환경, person 중심 추출 | 25W |
| 전력 효율 중심 실시간 추출 | 25W |
| 군중 장면 | MAXN_SUPER |
| 모든 클래스 추적 | MAXN_SUPER |
| Preview 또는 추가 출력 기능 확장 | MAXN_SUPER |

---

## 13. Thermal Note

고밀도 반복 테스트 이후 Jetson 방열판과 장치가 매우 뜨거워졌다.

현재까지 thermal throttling 여부를 정량적으로 확인하지 않았으므로, 장시간 안정성 테스트 전에는 성능 수치를 최종 확정하지 않는다.

다음 측정부터는 `tegrastats` 로그를 함께 기록한다.

확인 항목:

```text
CPU usage
GPU usage
RAM usage
Temperature
Power draw
Clock changes
Thermal throttling
FPS change over time
```

---

## 14. How to Run

### 14.1 GT State JSON

```bash
python src/mot17_gt_to_state_json.py
```

### 14.2 Original YOLO + ByteTrack State JSON

```bash
python src/yolo_bytetrack_to_state_json.py
```

### 14.3 Stage Timing

```bash
python experiments/timing/mot17_trt_stage_timing.py
```

### 14.4 Queue(maxsize=4) Benchmark

```bash
python experiments/cpu_queue/mot17_trt_async_reader.py
```

### 14.5 Sequence Stress Test

MOT17-04-DPM, person only:

```bash
MOT_SEQUENCE=MOT17-04-DPM \
STRESS_CLASSES=person \
python experiments/cpu_queue/mot17_trt_async_reader_sequence_stress.py
```

MOT17-03-DPM, all classes:

```bash
MOT_SEQUENCE=MOT17-03-DPM \
STRESS_CLASSES=all \
python experiments/cpu_queue/mot17_trt_async_reader_sequence_stress.py
```

---

## 15. Next Steps

### Phase 1. Thermal and Resource Logging

```text
tegrastats 로그 수집
→ 10분 테스트
→ 30분 테스트
→ 온도·전력·FPS 추이 확인
→ thermal throttling 여부 확인
```

### Phase 2. CSI Camera Live Input

```text
CSI Camera
→ GStreamer / camera input
→ limited queue
→ TensorRT YOLO11n
→ ByteTrack
→ State JSON
```

카메라 입력에서는 Queue가 가득 찰 때 오래된 프레임을 버리고 최신 프레임을 우선하는 정책을 검토한다.

### Phase 3. Lightweight Preview

```text
State
→ bbox + ID preview
→ FPS / object count display
```

trajectory, velocity arrow, mp4 저장은 선택 옵션 또는 후처리로 둔다.

### Phase 4. Power and Parameter Comparison

```text
15W / 25W / MAXN_SUPER
resolution
confidence threshold
queue size = 1 / 2 / 4 / 8
```

### Phase 5. Real-world Stress Test

```text
CSI 카메라
→ 사람이 적은 장면
→ 일반 장면
→ 혼잡 장면
→ 장시간 안정성 확인
```

---

## 16. Current Status Summary

```text
[완료] GT Label Replay
[완료] YOLO + ByteTrack Automatic Replay
[완료] Jetson TensorRT FP16 Extraction
[완료] 구간별 timing 로그
[완료] 전체 preload 방식의 부적합성 확인
[완료] Queue(maxsize=4) 기반 CPU 입력 최적화
[완료] 25W / MAXN_SUPER 3회 반복 비교
[완료] MOT17-02 / 04 / 03 객체 밀도 스트레스 테스트
[확인] 25W 최악 조건 평균 30.795 FPS
[확인] MAXN_SUPER 최악 조건 37.750 FPS
[다음] thermal throttling 및 자원 로그 측정
[다음] CSI 카메라 실시간 입력 연결
```

## 17. Benchmark Update — 2026-06-05

이 섹션은 기존 Queue 기반 quick benchmark 이후에 수행한 추가 측정 결과를 정리한다.

기존 결과를 삭제하지 않고 유지하되, 아래 수치를 입력 크기·confidence·리소스 비교를 포함한 최신 기준값으로 사용한다.

---

### 17.1 Current Baseline and Balance Candidate

공통 조건:

```text
Dataset        MOT17-02-DPM
Frames         600
Source image   1920 × 1080 JPG sequence
Model          YOLO11n
Runtime        TensorRT FP16
Tracker        ByteTrack
Queue          async queue size 4
Power mode     MAXN_SUPER
jetson_clocks  enabled

현재 용도별 권장 설정:

Purpose	Input size	Confidence	Reason
최고 속도	640	0.25	약 46 FPS, 추가 기능을 붙일 여유가 큼
실시간 균형형	960	0.25	약 36 FPS, 탐지 상태 수 증가
정확도 우선 오프라인	1280	0.25	탐지량은 높지만 약 21 FPS

현재 실시간 균형 후보:

YOLO11n TensorRT FP16
imgsz = 960
confidence = 0.25
ByteTrack
Queue(maxsize=4)
MAXN_SUPER
17.2 Consumer Stage Breakdown

MAXN_SUPER, YOLO11n TensorRT FP16, input size 640, confidence 0.15 조건에서 Consumer 내부 시간을 세분화하였다.

Stage	Avg time	Ratio
Preprocess	5.764 ms	28.0%
TensorRT YOLO inference	4.682 ms	22.8%
Postprocess	3.682 ms	17.9%
ByteTrack update	4.720 ms	23.0%
Framework overhead	1.719 ms	8.4%
Consumer total	20.566 ms	100%

핵심 관찰:

TensorRT YOLO inference time
≈
CPU-based ByteTrack update time

TensorRT FP16을 적용한 이후에는 GPU 추론만이 아니라 CPU 전처리, bbox 후처리, ByteTrack 비용도 전체 처리량에 큰 영향을 준다.

17.3 Power Mode Benchmark with tegrastats

동일한 YOLO11n TensorRT FP16 엔진을 사용하고 nvpmodel 전력 모드만 변경하였다.

각 조건은 3회 반복 측정하였다.

Power mode	Pipeline FPS	Consumer FPS	Decode ms	Infer + Track ms	Avg power	Avg CPU temp	Avg GPU temp
25W	37.518 ± 0.388	39.353 ± 0.417	15.628	23.611	7.430 W	52.949°C	52.809°C
MAXN_SUPER	45.275 ± 0.205	47.605 ± 0.234	12.335	19.697	8.202 W	53.357°C	53.198°C

해석:

MAXN_SUPER
→ 평균 전력 약 10.4% 증가
→ Pipeline FPS 약 20.7% 증가

단기 반복 구간에서는 명확한 thermal throttling이나 불안정성은 관찰되지 않았다.

17.4 Input Size Benchmark

원본 JPG는 항상 1920 × 1080으로 유지하였다.

TensorRT 엔진 내부 입력 크기만 640, 960, 1280으로 변경하였다.

공통 조건:

YOLO11n TensorRT FP16
MAXN_SUPER
ByteTrack
confidence = 0.15
Queue(maxsize=4)

각 조건은 3회 반복 측정하였다.

Input size	Pipeline FPS	Consumer FPS	Infer + Track ms	Objects	Pred / GT ratio	Avg power	Avg temp
640	45.778 ± 0.314	48.140 ± 0.362	19.432	5,377	0.2894	8.239 W	55.911°C
960	35.772 ± 0.304	37.034 ± 0.312	25.059	7,685	0.4136	9.274 W	56.069°C
1280	21.290 ± 0.135	21.733 ± 0.140	43.898	9,760	0.5253	9.723 W	56.427°C

해석:

640
→ 최고 속도 조건
→ 실시간 처리 여유가 큼

960
→ 30 FPS 이상 유지
→ 640 대비 객체 상태 수 약 42.9% 증가
→ 속도와 탐지 밀도의 균형 후보

1280
→ 객체 상태 수는 가장 많음
→ 약 21 FPS로 감소
→ 정확도 우선 오프라인 분석 조건

입력 크기 증가에 따라 JPG decode 시간은 거의 변하지 않았다.

640  : 12.372 ms
960  : 12.360 ms
1280 : 12.291 ms

따라서 속도 저하는 저장소 접근보다 TensorRT 추론, bbox 후처리, ByteTrack 부담 증가의 영향으로 해석할 수 있다.

17.5 Confidence Threshold Benchmark

YOLO11n TensorRT FP16, imgsz=960, MAXN_SUPER 조건에서 confidence threshold를 비교하였다.

Confidence	Pipeline FPS	Consumer FPS	Infer + Track ms	Objects	Unique tracks
0.15	34.984 ± 1.616	36.099 ± 1.817	25.846	7,685	176
0.25	36.585 ± 0.557	37.844 ± 0.585	24.438	7,685	176

최종 State JSON과 Replay의 객체 수 차이는 거의 없었다.

TensorRT 엔진에 confidence threshold가 고정된 것인지 확인하기 위해 순수 YOLO predict를 단일 프레임에서 추가 실행하였다.

Confidence	YOLO detections	Minimum confidence	Detections below 0.25
0.15	40	0.1530	14
0.25	26	0.2561	0

해석:

TensorRT 엔진에 confidence=0.25가 고정된 것은 아님.

confidence=0.15에서는
낮은 confidence bbox 후보가 실제로 증가함.

그러나 기본 ByteTrack 설정에서는
낮은 confidence 후보가 최종 active track으로 거의 남지 않음.

따라서 현재 기본 confidence는 0.25로 설정.

향후 tracker threshold를 함께 조절하면 낮은 confidence 객체의 활용 가능성을 별도로 검토할 수 있다.

17.6 TensorRT Engine Organization

입력 크기별 정적 TensorRT FP16 엔진을 생성하였다.

models/engines/
├─ yolo11n_trt_fp16_imgsz640_static.engine
├─ yolo11n_trt_fp16_imgsz960_static.engine
└─ yolo11n_trt_fp16_imgsz1280_static.engine

기존 실행 스크립트와의 호환성을 유지하기 위해 프로젝트 루트의 yolo11n.engine은 현재 활성 엔진을 가리키는 symbolic link로 사용한다.

기본 복구 상태:

yolo11n.engine
→ models/engines/yolo11n_trt_fp16_imgsz640_static.engine
17.7 Updated Status Summary
[완료] GT Label Replay
[완료] YOLO + ByteTrack Automatic Replay
[완료] Jetson CUDA 정상화
[완료] TensorRT FP16 엔진 생성
[완료] Queue(maxsize=4) 기반 CPU 입력 최적화
[완료] Consumer 내부 병목 분해
[완료] 25W / MAXN_SUPER 3회 반복 및 tegrastats 비교
[완료] input size 640 / 960 / 1280 비교
[완료] confidence 0.15 / 0.25 비교
[완료] 순수 YOLO predict를 통한 confidence 동작 확인
[확인] 실시간 균형 후보: imgsz=960, conf=0.25, MAXN_SUPER
[다음] YOLO11n vs YOLO11s 비교
[다음] Replay complexity benchmark
[다음] 최종 후보 5~10분 안정성 테스트
[후속] CSI 카메라 실시간 입력
[후속] tracker threshold 최적화
[후속] GPU tracker 또는 DeepStream nvtracker 검토
17.8 Next Experiment

다음 실험은 모델 크기 비교이다.

YOLO11n
vs
YOLO11s

비교 조건:

input size     960
confidence     0.25
tracker        ByteTrack
queue size     4
power mode     MAXN_SUPER
jetson_clocks  enabled
'''
