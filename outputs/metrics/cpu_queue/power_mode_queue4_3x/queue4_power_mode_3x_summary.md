# Queue(maxsize=4) Power Mode Benchmark

YOLO11n TensorRT FP16 + ByteTrack + State build, MOT17 JPG input.

## Aggregate Summary

| Mode | Runs | Pipeline FPS mean | Std | Min | Max | Decode ms | Infer+track ms | State ms | Consumer ms | P95 consumer ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25w | 3 | 37.871 | 0.236 | 37.656 | 38.124 | 15.646 | 23.546 | 1.596 | 25.146 | 27.632 |
| maxn_super | 3 | 45.448 | 0.144 | 45.292 | 45.575 | 12.377 | 19.558 | 1.395 | 20.957 | 23.241 |

## Raw Runs

| Mode | Run | Pipeline FPS | Elapsed sec | Decode ms | Queue wait ms | Infer+track ms | State ms | Consumer ms | P95 consumer ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25w | 1 | 37.656 | 15.9339 | 15.619 | 0.026 | 23.69 | 1.562 | 25.257 | 27.71 |
| 25w | 2 | 38.124 | 15.7383 | 15.626 | 0.026 | 23.363 | 1.637 | 25.004 | 27.445 |
| 25w | 3 | 37.832 | 15.8598 | 15.694 | 0.025 | 23.584 | 1.589 | 25.178 | 27.74 |
| maxn_super | 1 | 45.575 | 13.165 | 12.352 | 0.024 | 19.516 | 1.373 | 20.892 | 23.138 |
| maxn_super | 2 | 45.477 | 13.1934 | 12.396 | 0.023 | 19.68 | 1.257 | 20.941 | 23.192 |
| maxn_super | 3 | 45.292 | 13.2473 | 12.384 | 0.024 | 19.479 | 1.554 | 21.037 | 23.394 |
