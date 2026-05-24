# Jetson State Replay Project

## 1. Project Goal

이 프로젝트는 영상에서 객체의 상태를 추출하여 `State JSON`으로 구조화하고, 이 JSON을 기반으로 OpenCV 2D Replay를 생성하는 것을 목표로 한다.

최종적으로는 **Jetson Orin Nano Super** 환경에서 객체 탐지, 추적, State JSON 생성, 2D Replay 생성 과정을 실행하고, 처리 속도와 리소스 사용량을 측정한다.

```text
Video / Image Sequence
→ Object Detection
→ Object Tracking
→ State JSON
→ OpenCV 2D Replay
→ Performance Measurement
```

핵심 연구 방향은 단순히 YOLO를 실행하는 것이 아니라, **Jetson Orin Nano Super에서 영상 상태 추출 파이프라인이 어느 정도까지 실시간 처리 가능한지**를 실험적으로 확인하는 것이다.

---

## 2. Current MVP Status

현재 PC 환경에서 MVP-0, MVP-1 단계까지 구현하였다.

### MVP-0. GT Label Based State Replay

MOT17 정답 라벨인 `gt.txt`를 State JSON으로 변환한 뒤, OpenCV 기반 2D Replay 영상을 생성하였다.

```text
MOT17 gt.txt
→ State JSON
→ Overlay Replay
→ Abstract Replay
```

### MVP-1. YOLO + ByteTrack Automatic State Replay

MOT17 이미지 시퀀스를 YOLO11 + ByteTrack으로 처리하여 예측 State JSON을 생성하고, 동일한 OpenCV Replay 파이프라인으로 시각화하였다.

```text
MOT17 img1 frames
→ YOLO11 Object Detection
→ ByteTrack Object Tracking
→ State JSON
→ Overlay Replay
→ Abstract Replay
```

현재까지의 핵심 성과는 **정답 라벨 기반 Replay와 AI 자동 추출 기반 Replay를 모두 구현하고, 두 결과를 시각적/수치적으로 비교할 수 있게 만든 것**이다.

---

## 3. Dataset

현재 MVP 실험에는 MOT17 데이터셋의 `MOT17-02-DPM` 시퀀스를 사용하였다.

```text
data/
└─ MOT17-02-DPM/
   ├─ img1/
   │  ├─ 000001.jpg
   │  ├─ 000002.jpg
   │  └─ ...
   ├─ gt/
   │  └─ gt.txt
   └─ seqinfo.ini
```

시퀀스 정보는 다음과 같다.

| Item | Value |
|---|---:|
| Sequence | MOT17-02-DPM |
| Frame count | 600 |
| Source FPS | 30 |
| Resolution | 1920 x 1080 |
| Image extension | .jpg |

---

## 4. Dataset Policy

MOT17 원본 이미지 데이터는 GitHub repository에 포함하지 않는다.

GitHub에는 코드, README, 작은 요약 결과 파일만 저장한다.  
MOT17 데이터셋은 별도로 다운로드하거나 Google Drive를 통해 개인/팀 내부 공유용으로 관리한다.

데이터셋은 아래 경로에 배치해야 한다.

```text
jetson_state_replay/
└─ data/
   └─ MOT17-02-DPM/
      ├─ img1/
      ├─ gt/
      │  └─ gt.txt
      └─ seqinfo.ini
```

---

## 5. Folder Structure

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
├─ README.md
└─ requirements.txt
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

주요 필드는 다음과 같다.

| Field | Meaning |
|---|---|
| frame | 프레임 번호 |
| timestamp | 영상 시간 |
| id | 객체 추적 ID |
| class_name | 객체 클래스 이름 |
| confidence | YOLO 예측 신뢰도 |
| bbox_xywh | x, y, width, height |
| bbox_xyxy | x1, y1, x2, y2 |
| center | 객체 중심 좌표 |
| velocity | smoothing된 중심 이동 벡터 |
| raw_velocity | 원본 중심 이동 벡터 |
| source | GT 또는 YOLO+ByteTrack 출처 |

---

## 7. Replay Modes

현재 OpenCV Replay는 두 가지 방식으로 생성된다.

### 7.1 Overlay Replay

원본 영상 위에 bbox, object id, center point, velocity arrow, trajectory를 표시한다.

목적:

- 라벨 또는 예측 결과가 원본 영상 위에서 잘 맞는지 확인
- GT와 YOLO+ByteTrack 결과를 시각적으로 비교
- 탐지 누락, 병합, ID 변경 등을 디버깅

### 7.2 Abstract Replay

검은 배경 위에 객체 상태만 표시한다.

목적:

- 원본 영상 없이 State JSON만으로 2D 가상 장면 재현
- 객체 이동 경로와 추적 ID 확인
- 상태 기반 시뮬레이션의 MVP 결과물 생성

현재 `abstract` 파일은 본 프로젝트에서 말하는 **OpenCV 기반 2D State Replay 결과**로 볼 수 있다.  
다만 이는 완성형 3D 가상세계가 아니라, bbox, center, velocity, trajectory를 기반으로 한 **2D 상태 재현 MVP**이다.

---

## 8. Generated Output Files

### GT Label Based Replay

```text
outputs/state_json/mot17_02_gt_state.json
outputs/replay/mot17_02_gt_overlay.mp4
outputs/replay/mot17_02_gt_abstract.mp4
outputs/metrics/mot17_02_gt_replay_metrics.json
```

### YOLO + ByteTrack Based Replay

```text
outputs/state_json/mot17_02_pred_yolo_bytetrack_state.json
outputs/replay/mot17_02_pred_yolo_bytetrack_overlay.mp4
outputs/replay/mot17_02_pred_yolo_bytetrack_abstract.mp4
outputs/metrics/mot17_02_pred_yolo_bytetrack_metrics.json
outputs/metrics/mot17_02_pred_yolo_bytetrack_replay_metrics.json
```

### GT vs Prediction Summary

```text
outputs/metrics/mot17_02_gt_vs_pred_summary.json
outputs/metrics/mot17_02_gt_vs_pred_summary.csv
outputs/metrics/mot17_02_gt_vs_pred_summary.md
```

---

## 9. Current Quantitative Result

현재 MOT17-02-DPM 600프레임 기준 결과는 다음과 같다.

| Item | GT | YOLO+ByteTrack |
|---|---:|---:|
| Frame count | 600 | 600 |
| Total objects | 18,581 | 5,052 |
| Avg objects/frame | 30.968 | 8.42 |
| Max objects/frame | 36 | 16 |
| Unique track IDs | 62 | 114 |
| Conversion elapsed sec | - | 59.7279 |
| Estimated extraction FPS | - | 10.046 |
| Pred / GT total object ratio | - | 0.2719 |

현재 summary는 GT와 YOLO+ByteTrack 결과의 객체 수, 평균 객체 수, 추적 ID 수, 처리 FPS를 비교한 **1차 정량 요약**이다.

아직 Precision, Recall, MOTA, IDF1과 같은 정식 tracking metric은 계산하지 않았다.  
다음 단계에서 GT bbox와 prediction bbox를 IoU 기준으로 매칭하여 정확도 평가를 추가할 수 있다.

---

## 10. Key Observations

1. GT labels are much denser.
   - GT는 원거리, 작은 사람, 부분적으로 가려진 사람까지 촘촘하게 포함한다.

2. YOLO+ByteTrack mainly detects visually clear objects.
   - 가까운 객체, 선명한 객체, 전경에 있는 객체는 안정적으로 탐지한다.
   - 멀리 있는 사람, 겹친 사람, 가려진 사람은 누락되거나 하나로 병합되는 경향이 있다.

3. YOLO+ByteTrack extracts about 27.2% of GT object states.
   - 이는 단순한 버그라기보다 GT 라벨링 기준과 일반 객체 탐지 모델의 인식 기준 차이를 보여준다.

4. YOLO+ByteTrack generated more unique IDs than GT.
   - GT unique ID는 62개, YOLO+ByteTrack unique ID는 114개이다.
   - 전체 객체 수는 적지만 ID 수가 더 많다는 것은 일부 객체에서 ID가 끊기거나 새 ID로 재할당되는 tracking instability 가능성을 의미한다.

5. Bicycle-riding people can be treated differently.
   - MOT17 GT는 pedestrian tracking 대상 기준이다.
   - YOLO는 이미지에서 사람 형태가 보이면 `person`으로 탐지할 수 있다.

---

## 11. Baseline PC Environment

초기 MVP 파이프라인은 Samsung Galaxy Book Ion2 노트북에서 실행하였다.

| Item | Value |
|---|---|
| Device | Samsung Galaxy Book Ion2 |
| Model | 950XDA |
| RAM | 8GB |
| GPU | NVIDIA GeForce MX450 + Intel Iris Xe Graphics |
| OS | Windows |
| CPU | Intel Core i7 series |

CPU의 정확한 모델명은 추가 확인 후 기록한다.

이 PC 결과는 Jetson Orin Nano Super 성능 측정의 baseline으로 사용한다.  
단순히 PC와 Jetson의 FPS만 비교하는 것이 아니라, Jetson의 전력, 온도, 안정성, 실시간 처리 가능성을 함께 측정한다.

---

## 12. How to Run

### 12.1 GT State JSON 생성

```bash
python ./src/mot17_gt_to_state_json.py
```

### 12.2 YOLO + ByteTrack State JSON 생성

```bash
python ./src/yolo_bytetrack_to_state_json.py
```

### 12.3 2D Replay 생성

`replay_2d_from_state.py` 내부의 `STATE_PATH`와 출력 파일명을 GT 또는 Prediction에 맞게 설정한 뒤 실행한다.

```bash
python ./src/replay_2d_from_state.py
```

### 12.4 GT vs Pred Summary 생성

```bash
python ./src/make_gt_pred_summary.py
```

출력:

```text
outputs/metrics/mot17_02_gt_vs_pred_summary.json
outputs/metrics/mot17_02_gt_vs_pred_summary.csv
outputs/metrics/mot17_02_gt_vs_pred_summary.md
```

---

## 13. Jetson Transfer Plan

Jetson Orin Nano Super에서는 GitHub repository를 clone한 뒤, MOT17-02-DPM 데이터셋만 별도로 `data/` 폴더에 배치한다.

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/moongun515/jetson_state_replay.git
cd jetson_state_replay
mkdir -p data
```

그다음 Google Drive 또는 USB로 받은 `MOT17-02-DPM` 폴더를 아래 위치에 둔다.

```text
~/projects/jetson_state_replay/data/MOT17-02-DPM/
```

이후 Jetson에서 동일한 파이프라인을 실행하여 PC baseline과 FPS, latency, replay time, 리소스 사용량을 비교한다.

---

## 14. Jetson Performance Measurement Plan

Jetson에서의 핵심 목적은 **YOLO11 + ByteTrack + 2D Replay 파이프라인이 엣지 장비에서 어느 정도까지 처리 가능한지**를 확인하는 것이다.

### 14.1 Measurement Metrics

| Category | Metric |
|---|---|
| Speed | YOLO inference FPS |
| Speed | Tracking FPS |
| Speed | End-to-End FPS |
| Latency | Inference latency |
| Latency | End-to-End latency |
| Resource | GPU usage |
| Resource | CPU usage |
| Resource | RAM usage |
| Resource | Temperature |
| Resource | Power consumption |
| Stability | Frame drop rate |
| Stability | Thermal throttling |

### 14.2 Experiment Variables

#### Variable 1. Input size / resolution

```text
imgsz 640
imgsz 960
imgsz 1280
```

또는 실제 프레임 리사이즈 기준:

```text
480p
720p
1080p
```

#### Variable 2. Model size

```text
YOLO11n
YOLO11s
YOLO11m
```

Jetson Orin Nano Super에서는 우선 `YOLO11n`, `YOLO11s`를 중심으로 측정하고, `YOLO11m`은 한계 확인용으로 사용한다.

#### Variable 3. Replay complexity

```text
Replay off
bbox only
bbox + trajectory + velocity
overlay + abstract replay
```

3D/Unity Replay는 바로 구현하지 않고, Jetson 성능 측정 이후의 확장 실험으로 둔다.

---

## 15. Recommended Experiment Order

### Phase 1. Fixed Video Baseline

고정 입력인 MOT17-02-DPM 600프레임을 사용해 PC와 Jetson의 baseline 성능을 비교한다.

```text
MOT17-02-DPM
YOLO11n
imgsz 640
Replay off / bbox replay
```

### Phase 2. Three Variable Test

Jetson에서 세 가지 변수를 조절하며 성능 변화를 측정한다.

```text
A. Input size: 640 / 960 / 1280
B. Model size: YOLO11n / YOLO11s
C. Replay complexity: off / bbox only / bbox + trajectory
```

### Phase 3. Real Video Test

고정 데이터셋 기준 성능을 확인한 뒤, 실제 촬영 영상 또는 카메라 입력 영상으로 일반화 테스트를 진행한다.

```text
Jetson camera or recorded video
→ YOLO11 + ByteTrack
→ State JSON
→ 2D Replay
→ Performance measurement
```

---

## 16. Future Work

1. GT bbox와 Prediction bbox를 IoU 기준으로 매칭하여 Precision, Recall 계산
2. Tracking metric인 ID Switch, MOTA, IDF1 계산
3. Jetson tegrastats 로그 기반 성능표 작성
4. 해상도, 모델 크기, Replay 복잡도별 성능 저하 분석
5. 실제 촬영 영상 기반 일반화 테스트
6. Side-by-side GT vs Prediction Abstract Replay 생성
7. 3D/Unity Replay 확장 실험

---

## 17. Current Conclusion

현재 프로젝트는 PC 환경에서 **State JSON 생성, OpenCV 2D Replay, GT vs YOLO+ByteTrack 1차 정량 비교**까지 완료한 MVP 단계이다.

다음 핵심 단계는 Jetson Orin Nano Super에서 동일한 파이프라인을 실행하고, 해상도, 모델 크기, Replay 복잡도 변화에 따른 성능 한계를 측정하는 것이다.

현재 abstract replay는 본 프로젝트의 2D 시뮬레이션 MVP 결과물이며, 이후 Jetson 성능 측정 결과와 결합하면 영상 상태 추출 기반 가상 장면 재현 프로젝트의 실험적 가치가 더 명확해진다.

---

## 18. Jetson Orin Nano Super First Runtime Result

Jetson Orin Nano Super에서 MOT17-02-DPM 600프레임을 사용하여 1차 실행 테스트를 완료하였다.

### 18.1 Environment Note

현재 Jetson의 기존 `yolo_env` 가상환경을 사용하였다.

- ultralytics: 8.4.46
- torch: 2.11.0+cu130
- OpenCV: 4.13.0
- NumPy: 1.24.4
- SciPy: 1.10.1
- CUDA available: False

현재 PyTorch CUDA 버전과 Jetson 드라이버/CUDA 버전이 맞지 않아 GPU 가속이 적용되지 않았다.  
따라서 아래 YOLO+ByteTrack 결과는 Jetson GPU 성능이 아니라 **CPU fallback 상태의 1차 실행 결과**이다.

### 18.2 Jetson GT Replay Result

| Item | Value |
|---|---:|
| Input frames | 600 |
| Written frames | 600 |
| Total objects drawn | 18,581 |
| Avg objects/frame | 30.968 |
| Avg read time | 17.626 ms/frame |
| Avg render time | 59.393 ms/frame |
| Avg write time | 89.492 ms/frame |
| Avg frame time | 166.747 ms/frame |
| Processing FPS | 5.973 |

### 18.3 Jetson YOLO11n + ByteTrack Result

| Item | Value |
|---|---:|
| Model | YOLO11n |
| Input frames | 600 |
| Total tracked objects | 5,052 |
| Elapsed time | 287.5375 sec |
| Processing FPS | 2.087 |

### 18.4 Jetson Prediction Replay Result

| Item | Value |
|---|---:|
| Input frames | 600 |
| Written frames | 600 |
| Total objects drawn | 5,052 |
| Avg objects/frame | 8.420 |
| Avg read time | 17.388 ms/frame |
| Avg render time | 22.735 ms/frame |
| Avg write time | 70.039 ms/frame |
| Avg frame time | 110.403 ms/frame |
| Processing FPS | 9.032 |

### 18.5 Jetson GT vs Prediction Summary

| Item | GT | YOLO+ByteTrack |
|---|---:|---:|
| Frame count | 600 | 600 |
| Total objects | 18,581 | 5,052 |
| Avg objects/frame | 30.968 | 8.42 |
| Max objects/frame | 36 | 16 |
| Unique track IDs | 62 | 114 |
| Conversion elapsed sec | - | 287.5375 |
| Estimated extraction FPS | - | 2.087 |
| Pred / GT total object ratio | - | 0.2719 |

### 18.6 Current Interpretation

Jetson에서 GT 기반 State JSON 생성, GT 2D Replay, YOLO11n + ByteTrack 기반 Pred State JSON 생성, Pred 2D Replay까지 모두 성공하였다.

다만 현재 YOLO+ByteTrack은 CUDA를 사용하지 못하고 CPU fallback으로 실행되었기 때문에 처리 속도가 2.087 FPS로 낮게 측정되었다.  
이는 Jetson Orin Nano Super의 실제 GPU 가속 성능이 아니라, 환경 호환성 확인용 1차 실행 결과이다.

Replay 단계에서는 GT보다 Prediction 결과가 더 빠르게 처리되었다.  
이는 GT가 18,581개의 객체 상태를 그리는 반면, YOLO+ByteTrack Prediction은 5,052개의 객체 상태만 그리기 때문이다.  
따라서 객체 수가 Replay 렌더링 성능에 직접적인 영향을 준다는 것을 확인할 수 있다.

다음 단계는 Jetson에 맞는 PyTorch/CUDA 또는 TensorRT 환경을 구성하여 GPU 가속 상태에서 동일한 실험을 다시 수행하는 것이다.
