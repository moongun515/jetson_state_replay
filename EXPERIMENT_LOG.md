# Jetson Orin Nano Super Experiment Log

## Baseline

- Dataset: MOT17-02-DPM
- Frames: 600
- Model: YOLO11n
- Runtime: TensorRT FP16
- Engine: yolo11n.engine
- Tracker: ByteTrack
- Confidence: 0.15
- Queue size: 4
- Queue policy: preserve_all_frames_block_when_full
- Power mode: MAXN_SUPER
- jetson_clocks: enabled
- Warmup: exclude first 30 frames or run one warmup trial

## Current Baseline Result

- Pipeline FPS: approximately 45-46
- Consumer FPS: approximately 48
- Decode: approximately 12.4 ms/frame
- Queue wait: approximately 0.02 ms/frame
- Consumer total: approximately 20.6 ms/frame

## Consumer Breakdown

| Stage | Avg time |
|---|---:|
| Preprocess | 5.764 ms |
| TensorRT YOLO inference | 4.682 ms |
| Postprocess | 3.682 ms |
| ByteTrack update | 4.720 ms |
| Framework overhead | 1.719 ms |
| Consumer total | 20.566 ms |

## Thermal Quick Check

- MAXN_SUPER continuous 3 runs
- Temperature increased from approximately 45 C to 53 C
- No clear FPS degradation observed
- Long stability test remains pending

## Planned Experiments

### A. Power Mode

| Condition | Runs | Status |
|---|---:|---|
| 25W | 3 | pending |
| MAXN_SUPER | 3 | pending |

### B. Input Size

| Condition | Runs | Status |
|---|---:|---|
| 640 | 3 | pending |
| 960 | 3 | pending |
| 1280 | 3 | pending |

### C. Model Size

| Condition | Runs | Status |
|---|---:|---|
| YOLO11n | 3 | pending |
| YOLO11s | 3 | pending |

### D. Confidence Threshold

| Condition | Runs | Status |
|---|---:|---|
| 0.15 | 3 | pending |
| 0.25 | 3 | pending |

### E. Replay Complexity

| Condition | Runs | Status |
|---|---:|---|
| bbox only | 3 | pending |
| bbox + trajectory + velocity | 3 | pending |
| overlay + abstract | 3 | pending |

### F. Long Stability

| Condition | Duration | Status |
|---|---:|---|
| Final optimized configuration | 5-10 min | pending |

## Future Research

- CSI camera real-time input
- GStreamer / nvarguscamerasrc
- Low-latency queue policy
- GPU tracker or DeepStream nvtracker
- Multi-camera
- 3D-like replay

## 2026-06-05 Experiment Summary

### Power mode benchmark

| Mode | Pipeline FPS | Consumer FPS | Avg power |
|---|---:|---:|---:|
| 25W | 37.518 | 39.353 | 7.430 W |
| MAXN_SUPER | 45.275 | 47.605 | 8.202 W |

- MAXN_SUPER improved Pipeline FPS by approximately 20.7%.
- No clear short-term thermal throttling was observed.

### Consumer breakdown at imgsz=640

| Stage | Avg time |
|---|---:|
| Preprocess | 5.764 ms |
| TensorRT YOLO inference | 4.682 ms |
| Postprocess | 3.682 ms |
| ByteTrack update | 4.720 ms |
| Framework overhead | 1.719 ms |
| Consumer total | 20.566 ms |

### Input size benchmark at MAXN_SUPER

| Input size | Pipeline FPS | Objects | Pred / GT ratio |
|---:|---:|---:|---:|
| 640 | 45.778 | 5377 | 0.2894 |
| 960 | 35.772 | 7685 | 0.4136 |
| 1280 | 21.290 | 9760 | 0.5253 |

- 960 is the current balance candidate for real-time processing.
- 1280 improves object density but falls below the 30 FPS target.

### Confidence benchmark at imgsz=960

| Confidence | Pipeline FPS | Objects |
|---:|---:|---:|
| 0.15 | 34.984 | 7685 |
| 0.25 | 36.585 | 7685 |

Pure YOLO predict check on a sample frame:

| Confidence | YOLO detections | Detections below 0.25 |
|---:|---:|---:|
| 0.15 | 40 | 14 |
| 0.25 | 26 | 0 |

Interpretation:
- TensorRT does not hard-code confidence=0.25.
- Lower-confidence YOLO candidates exist at conf=0.15.
- Under the default ByteTrack configuration, these candidates rarely survive into final active tracks.
- Default operating confidence is set to 0.25 for now.

### Current recommended balance configuration

- Model: YOLO11n
- Runtime: TensorRT FP16
- Input size: 960
- Confidence: 0.25
- Tracker: ByteTrack
- Queue size: 4
- Power mode: MAXN_SUPER

### Next experiments

1. YOLO11n vs YOLO11s comparison at imgsz=960 and confidence=0.25
2. Replay complexity benchmark
3. Final 5-10 minute stability test
4. Future extension: CSI camera real-time input and tracker threshold optimization
