# FallGuard Android 시제품

이 앱은 1~30초짜리 영상을 사용자가 직접 고르면 YOLO26n-pose와 관절 기반 낙상 분류기를 **휴대폰 안에서** 실행한다. `INTERNET` 권한, 서버 주소, 영상 업로드 기능이 없다. 영상 선택 창에는 로컬 파일만 요청한다. 외부 클라우드 앱에서 공유한 영상을 열면 해당 앱의 저장 정책은 별개이므로, 엄격히 오프라인으로 쓰려면 기기에 저장한 영상을 선택한다. 실시간 카메라 경보와 의료·안전 용도는 아직 검증되지 않았다.

## 열기와 실행

1. Android Studio에서 `android/fallguard` 폴더를 연다.
2. `app/src/main/assets/yolo26n-pose.onnx` 파일이 있는지 확인한다. 로컬 개발 환경에는 이미 준비되어 있다. GitHub에는 큰 모델 파일을 올리지 않는다.
3. 휴대폰 또는 Android 에뮬레이터를 선택해 `app`을 실행한다. 디버그 APK는 `app/build/outputs/apk/debug/app-debug.apk`에 생성된다.
4. 앱의 **짧은 영상 선택** 버튼으로 1~30초 영상을 고른다. 12프레임을 뽑아 낙상/쓰러진 상태 확률을 표시한다.

2026-11-02 Android API 36 에뮬레이터에서 디버그 APK 설치와 영상 분석을 확인했다. 공개 CAUCAFall 시험 클립을 MP4로 바꿔 실행했을 때 앱 확률은 95.4%, 같은 MP4의 Python ONNX 확률은 95.85%였다. 이는 **한 영상의 실행 확인**이며 Android 전체 시험셋 정확도나 실제 휴대폰 지연시간 측정은 아니다. 디버그 APK는 약 91MB이며 패키지 권한 검사에서 `INTERNET` 권한이 없었다.

모델 파일을 다시 만들려면 Python 환경에 `ultralytics`, `onnx`를 설치한 뒤 아래를 실행한다. 로컬에서 검증한 Ultralytics 버전은 8.4.53, ONNX 버전은 1.23.2다. 출력은 `(1, 300, 57)`이어야 한다.
현재 검증한 ONNX 파일의 SHA-256은 `36f58b87f7d58cd6e98e11aeac4c1607bba866f40805933d352d4710ecf6ac48`이다.

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo26n-pose.pt').export(format='onnx', imgsz=640, nms=False, simplify=False, dynamic=False)"
Copy-Item yolo26n-pose.onnx android/fallguard/app/src/main/assets/yolo26n-pose.onnx
```

`classifier.json`은 [CAUCAFall 실험](../../fall_detection/README.md)의 학습 결과에서 만든다. 학습 보고서와 모델이 있으면 다음 명령으로 재생성할 수 있다.

```powershell
python fall_detection/export_android_classifier.py --report TRAINING_DIR/report.json --output android/fallguard/app/src/main/assets/classifier.json
```

## 해석할 때 주의할 점

- 현재 성능 수치는 사람을 분리한 CAUCAFall 시험 클립 55개에서 낙상 20개 중 19개 탐지, 비낙상 35개 중 3개 오경보다. 실제 집에서의 시간당 오경보율은 모른다.
- 정상적으로 누워 있는 장면은 전체 학습 자료에 1개뿐이고 시험에는 없다. 침대나 소파에서 눕는 행동은 별도 자료로 검증해야 한다.
- Android 영상 추출과 이미지 크기 조정은 Python/OpenCV와 완전히 같지 않을 수 있다. 기기에서 동일 클립을 비교한 정밀도 시험이 아직 필요하다.
- 현재 ONNX는 FP32다. INT8은 필수가 아니며, 충분한 보정 영상과 동일 조건 비교 후 선택한다.
- 디버그 APK는 모든 ABI용 ONNX Runtime을 포함해서 크다. 최종 배포 전 arm64 전용 빌드 및 런타임 경량화를 검토한다.
