# YOLO26 관절 기반 경량 낙상 감지 실험

목표는 `YOLO26n-pose`가 추출한 COCO-17 관절 시퀀스로 낙상 동작 또는 쓰러진 상태를 감지하는 작은 분류기를 만드는 것이다. KITTI의 사람 검출 AP40과 INT8 결과는 이 작업의 낙상 인식 정확도나 관절 추정 정확도를 증명하지 않는다.

## 데이터와 분리

- [CAUCAFall 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9508401/)의 RGB 클립을 [OmniFall 공개 구간 라벨](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/main/README.md)과 함께 사용한다. 원본 자료는 CC BY 4.0으로 안내되어 있다.
- Zenodo의 [CAUCAFall ZIP](https://zenodo.org/records/17170592)을 받아 게시 MD5 `5a1899fe0022c9c4c0d37ce64ac0c986`과 일치함을 확인했다. ZIP에는 같은 구간을 가리키는 두 이름의 AVI가 각 258개 있으므로, 코드에서는 CSV 258행과 일치하는 한 벌만 사용한다.
- 라벨 `1=fall`, `2=fallen`을 양성으로, 나머지를 음성으로 둔다. `lying`은 음성이지만 이 데이터에는 단 한 구간만 있고 시험 분할에는 없어 침대에 눕는 상황을 제대로 검증할 수 없다.
- 같은 사람의 클립이 분할을 넘지 않도록 subject 1–6을 train, 7–8을 validation, 9–10을 test에 고정한다. 테스트셋은 임계값과 하이퍼파라미터 선택에 쓰지 않는다.

## 학습 방법

1. `prepare_caucafall.py`가 사전학습된 YOLO26n-pose를 고정한 채 각 클립에서 12프레임의 관절·사람 박스 특징을 추출한다. 중간 결과는 클립별로 캐시한다.
2. `train_temporal.py`가 관절 시퀀스의 시간 통계로 L2 로지스틱 회귀 분류기를 학습한다. 정규화 강도와 경보 임계값은 validation에서 고른다.
3. 사람 단위로 격리한 test에서 낙상 precision, recall, F1, false alarm을 기록한다. 이 수치는 staged CAUCAFall의 결과이며 가정집·Jetson·휴대폰 현장 성능을 뜻하지 않는다.

## 2026-11-02 첫 실행 결과

- 258개 클립에서 관절을 추출했다. 12프레임씩 뽑은 총 3096프레임의 사람/관절 탐지율은 88.95%였다.
- 학습 153개 클립, 검증 50개 클립, 시험 55개 클립이며 각 분할에 같은 사람이 겹치지 않는다.
- 시험: 낙상 20개 중 19개 탐지, 비낙상 35개 중 3개 오경보. Precision 86.36%, recall 95.00%, F1 90.48%.
- 저장된 짧은 시험 영상 1개를 독립 추론하여 저장된 예측 확률 `0.949106652305993`을 재현했다.
- 분류기 크기 13,905바이트, YOLO26n-pose 체크포인트 크기 7,878,574바이트. 둘을 함께 사용한다.

가장 큰 빈틈은 `lying` 음성 예제가 학습 분할에 1개, 검증·시험 분할에 0개라는 점이다. 장면도 제한된 재연 낙상으로 구성되어 있다. 따라서 이 수치로 실제 생활의 오경보율이나 낙상 정확도를 보장할 수 없다. 시험 수치는 **짧게 잘라 둔 클립 단위**이며 시간당 오경보나 실시간 지연시간은 측정하지 않았다.

```powershell
python fall_detection/prepare_caucafall.py --data-root DATA_ROOT --model yolo26n-pose.pt --output pose_features.npz
python fall_detection/train_temporal.py --features pose_features.npz --output-dir fall_training_result
python fall_detection/infer_short_clip.py --video SHORT_CLIP.avi --pose-model yolo26n-pose.pt --classifier fall_training_result/fall_pose_classifier.joblib --threshold 0.5
```

추론기는 학습과 동일하게 전체 클립에서 12프레임을 골라 판정한다. 길이가 긴 영상이나 실시간 스트림에 바로 적용하기 전에 시간 창, 경보 유지 조건, 침대에서 눕는 장면을 포함한 추가 평가가 필요하다.

현재는 YOLO26 관절망을 추가 학습하지 않는다. CAUCAFall은 낙상 구간 라벨을 제공하지만, 사람의 17개 관절 정답 좌표를 제공하지 않기 때문이다. 이후 자체 관절 라벨과 더 다양한 낙상·비낙상 영상을 확보하면 pose fine-tuning과 INT8 양자화를 별도로 비교한다. INT8 비교에는 train 전용 영상으로 보정하고 동일한 test 클립에서 FP32/INT8의 낙상 recall, false alarm, 지연시간을 함께 측정해야 한다.

이 실험을 휴대폰에서 실행하는 오프라인 시제품은 [FallGuard Android](../android/fallguard/README.md)에 있다.
