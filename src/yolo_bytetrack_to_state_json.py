import json
import time
import configparser
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

SEQ_DIR = ROOT / "data" / "MOT17-02-DPM"
SEQINFO_PATH = SEQ_DIR / "seqinfo.ini"
IMG_DIR = SEQ_DIR / "img1"

OUT_DIR = ROOT / "outputs" / "state_json"
OUT_PATH = OUT_DIR / "mot17_02_pred_yolo_bytetrack_state.json"

METRICS_DIR = ROOT / "outputs" / "metrics"
METRICS_PATH = METRICS_DIR / "mot17_02_pred_yolo_bytetrack_metrics.json"

# 먼저 가장 가벼운 모델로 시작
MODEL_NAME = "yolo11s.pt"

# MVP에서는 person만 추적
TARGET_CLASS_NAMES = {"person"}

# 너무 낮게 잡으면 false positive가 늘 수 있음
CONF_THRESHOLD = 0.15

# 전체 600장 다 돌리기 전에 빠른 테스트하고 싶으면 60 등으로 설정
# 전체 실행은 None
MAX_FRAMES = None


def read_seqinfo(seqinfo_path: Path) -> dict:
    config = configparser.ConfigParser()
    config.read(seqinfo_path, encoding="utf-8")
    seq = config["Sequence"]

    return {
        "name": seq.get("name", "unknown"),
        "image_dir": str(IMG_DIR).replace("\\", "/"),
        "fps": seq.getint("frameRate"),
        "frame_count": seq.getint("seqLength"),
        "width": seq.getint("imWidth"),
        "height": seq.getint("imHeight"),
        "image_ext": seq.get("imExt", ".jpg"),
    }


def get_image_paths(img_dir: Path, image_ext: str):
    paths = sorted(img_dir.glob(f"*{image_ext}"))
    if MAX_FRAMES is not None:
        paths = paths[:MAX_FRAMES]
    return paths


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def convert_yolo_bytetrack_to_state():
    start_time = time.perf_counter()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    video_info = read_seqinfo(SEQINFO_PATH)
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    image_ext = video_info["image_ext"]

    image_paths = get_image_paths(IMG_DIR, image_ext)
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in {IMG_DIR}")

    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    frames = []
    last_center_by_id = {}
    last_smoothed_velocity_by_id = {}
    velocity_alpha = 0.7

    per_frame_metrics = []

    total_detected = 0
    total_tracked = 0
    skipped_no_id = 0

    for idx, img_path in enumerate(image_paths, start=1):
        frame_start = time.perf_counter()

        try:
            frame_idx = int(img_path.stem)
        except ValueError:
            frame_idx = idx

        read_start = time.perf_counter()
        img = imread_unicode(img_path)
        read_elapsed = time.perf_counter() - read_start

        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        track_start = time.perf_counter()

        # persist=True가 프레임 간 tracking 상태 유지에 중요
        results = model.track(
            source=img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        track_elapsed = time.perf_counter() - track_start

        objects = []

        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes

            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0].item()) if box.conf is not None else 0.0

                    total_detected += 1

                    if cls_name not in TARGET_CLASS_NAMES:
                        continue

                    # ByteTrack이 id를 못 붙인 detection은 State JSON에서 제외
                    if box.id is None:
                        skipped_no_id += 1
                        continue

                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1 = max(0.0, min(float(width), float(x1)))
                    y1 = max(0.0, min(float(height), float(y1)))
                    x2 = max(0.0, min(float(width), float(x2)))
                    y2 = max(0.0, min(float(height), float(y2)))

                    w = x2 - x1
                    h = y2 - y1
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    prev_center = last_center_by_id.get(track_id)
                    prev_smoothed_velocity = last_smoothed_velocity_by_id.get(track_id)

                    if prev_center is None:
                        raw_velocity = [0.0, 0.0]
                        smoothed_velocity = [0.0, 0.0]
                    else:
                        raw_velocity = [
                            center_x - prev_center[0],
                            center_y - prev_center[1],
                        ]

                        if prev_smoothed_velocity is None:
                            smoothed_velocity = raw_velocity
                        else:
                            smoothed_velocity = [
                                velocity_alpha * prev_smoothed_velocity[0]
                                + (1 - velocity_alpha) * raw_velocity[0],
                                velocity_alpha * prev_smoothed_velocity[1]
                                + (1 - velocity_alpha) * raw_velocity[1],
                            ]

                    last_center_by_id[track_id] = [center_x, center_y]
                    last_smoothed_velocity_by_id[track_id] = smoothed_velocity

                    obj = {
                        "id": track_id,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 5),
                        "bbox_xywh": [
                            round(x1, 3),
                            round(y1, 3),
                            round(w, 3),
                            round(h, 3),
                        ],
                        "bbox_xyxy": [
                            round(x1, 3),
                            round(y1, 3),
                            round(x2, 3),
                            round(y2, 3),
                        ],
                        "center": [
                            round(center_x, 3),
                            round(center_y, 3),
                        ],
                        "velocity": [
                            round(smoothed_velocity[0], 3),
                            round(smoothed_velocity[1], 3),
                        ],
                        "raw_velocity": [
                            round(raw_velocity[0], 3),
                            round(raw_velocity[1], 3),
                        ],
                        "visibility": None,
                        "source": "yolo_bytetrack",
                    }

                    objects.append(obj)
                    total_tracked += 1

        frame_elapsed = time.perf_counter() - frame_start

        frames.append(
            {
                "frame": frame_idx,
                "timestamp": round((frame_idx - 1) / fps, 6),
                "object_count": len(objects),
                "objects": objects,
            }
        )

        per_frame_metrics.append(
            {
                "frame": frame_idx,
                "object_count": len(objects),
                "read_time_ms": round(read_elapsed * 1000, 3),
                "track_time_ms": round(track_elapsed * 1000, 3),
                "total_frame_time_ms": round(frame_elapsed * 1000, 3),
            }
        )

        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(image_paths)} frames")

    elapsed = time.perf_counter() - start_time

    state = {
        "schema_version": "state_json_v1.1",
        "video_info": {
            **video_info,
            "used_frame_count": len(frames),
            "first_existing_frame": frames[0]["frame"] if frames else None,
            "last_existing_frame": frames[-1]["frame"] if frames else None,
            "source_type": "yolo_bytetrack_pred",
        },
        "model_info": {
            "detector": MODEL_NAME,
            "tracker": "bytetrack.yaml",
            "conf_threshold": CONF_THRESHOLD,
            "target_class_names": sorted(list(TARGET_CLASS_NAMES)),
            "note": "Pretrained YOLO detection + ByteTrack tracking. No training or fine-tuning in this MVP step.",
        },
        "conversion_summary": {
            "input_frames": len(image_paths),
            "used_frames": len(frames),
            "total_detected_boxes_before_class_filter": total_detected,
            "total_tracked_objects_after_filter": total_tracked,
            "skipped_no_track_id": skipped_no_id,
            "elapsed_sec": round(elapsed, 4),
            "estimated_processing_fps": round(len(frames) / elapsed, 3) if elapsed > 0 else 0,
        },
        "frames": frames,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    metrics = {
        "state_output": str(OUT_PATH).replace("\\", "/"),
        "model": MODEL_NAME,
        "tracker": "bytetrack.yaml",
        "input_frames": len(image_paths),
        "elapsed_sec": round(elapsed, 4),
        "estimated_processing_fps": round(len(frames) / elapsed, 3) if elapsed > 0 else 0,
        "total_detected_boxes_before_class_filter": total_detected,
        "total_tracked_objects_after_filter": total_tracked,
        "skipped_no_track_id": skipped_no_id,
        "per_frame_metrics": per_frame_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] Pred State JSON saved: {OUT_PATH}")
    print(f"[OK] Metrics saved: {METRICS_PATH}")
    print(f"[INFO] Frames: {len(frames)}")
    print(f"[INFO] Total tracked objects: {total_tracked}")
    print(f"[INFO] Elapsed: {elapsed:.4f} sec")
    print(f"[INFO] Processing FPS: {len(frames) / elapsed:.3f}")


if __name__ == "__main__":
    convert_yolo_bytetrack_to_state()