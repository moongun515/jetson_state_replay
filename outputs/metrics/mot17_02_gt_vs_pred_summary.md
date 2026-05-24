# MOT17-02 GT vs YOLO+ByteTrack Summary

## 1. Experiment Overview

- Dataset sequence: `MOT17-02-DPM`
- Input frames: 600
- Resolution: 1920 x 1080
- Source FPS: 30
- GT pipeline: `gt.txt → State JSON → 2D Replay`
- Prediction pipeline: `MOT17 images → YOLO11 + ByteTrack → State JSON → 2D Replay`

## 2. Quantitative Summary

| Item | GT | YOLO+ByteTrack |
|---|---:|---:|
| Frame count | 600 | 600 |
| Total objects | 18581 | 5052 |
| Avg objects/frame | 30.968 | 8.42 |
| Max objects/frame | 36 | 16 |
| Unique track IDs | 62 | 114 |
| Conversion elapsed sec | None | 287.5375 |
| Estimated extraction FPS | None | 2.087 |

## 3. Ratio Summary

| Metric | Value |
|---|---:|
| Pred / GT total objects ratio | 0.2719 |
| Pred / GT avg objects per frame ratio | 0.2719 |

## 4. Observations

- GT labels are much denser and include many small or distant pedestrians.
- YOLO+ByteTrack mainly detects clear, foreground, and visually confident person objects.
- Distant, small, occluded, or highly overlapping people are often missed or merged.
- Bicycle-riding people may still be detected as `person` by YOLO, even when MOT17 GT does not treat them as pedestrian tracking targets.
- This difference is not simply a bug; it shows the difference between dataset labeling policy and general object detection behavior.

## 5. Current MVP Status

- MVP-0 completed: GT label based State JSON and 2D Replay.
- MVP-1 completed: YOLO11 + ByteTrack based automatic State JSON and 2D Replay.
- Next step: compare conditions such as confidence threshold, model size, input image size, and Jetson runtime performance.
