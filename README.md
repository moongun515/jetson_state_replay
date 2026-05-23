# Jetson Orin Nano Super 영상 상태 추출 및 2D Replay 프로젝트

## 1. 프로젝트 목표
영상에서 객체 상태를 추출하고 State JSON으로 변환한 뒤, 2D 가상 장면으로 재현한다.

## 2. 전체 파이프라인
MOT17 / Camera Video
→ YOLO11 + ByteTrack
→ State JSON
→ OpenCV Replay
→ Overlay / Replay Video

## 3. 현재 구현 상태
- MOT17 GT 기반 State JSON 변환 완료
- GT 기반 OpenCV Replay 영상 생성 완료
- YOLO11 + ByteTrack 예측 State JSON 생성 완료
- 예측 기반 Replay 영상 생성 완료

## 4. 현재 결과물
| 파일 | 설명 |
|---|---|
| mot17_02_gt_overlay.mp4 | MOT17 정답 라벨 기반 overlay |
| mot17_02_pred_yolo_bytetrack_overlay.mp4 | YOLO11 + ByteTrack 예측 기반 overlay |

## 5. 1차 비교 요약
summary 표 삽입

## 6. 다음 실험 계획
- 정확도 비교
- Jetson 성능 측정
- 해상도/모델 크기/재현 복잡도별 한계 테스트