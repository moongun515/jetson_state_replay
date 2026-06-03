import csv
import json
import time
import configparser
import statistics
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Store this file at:
# ~/jetson_state_replay/experiments/timing/mot17_trt_preload_timing.py
ROOT = Path(__file__).resolve().parents[2]

SEQ_DIR = ROOT / "data" / "MOT17-02-DPM"
SEQINFO_PATH = SEQ_DIR / "seqinfo.ini"
IMG_DIR = SEQ_DIR / "img1"

OUT_DIR = ROOT / "outputs" / "state_json" / "final"
OUT_PATH = OUT_DIR / "mot17_02_yolo11n_trt_fp16_preload_state.json"

METRICS_DIR = ROOT / "outputs" / "metrics" / "final"
CSV_PATH = METRICS_DIR / "mot17_02_yolo11n_trt_fp16_preload_timing.csv"
SUMMARY_PATH = METRICS_DIR / "mot17_02_yolo11n_trt_fp16_preload_summary.json"

MODEL_NAME = "yolo11n.engine"
TARGET_CLASS_NAMES = {"person"}
CONF_THRESHOLD = 0.15

# Use None for all 600 frames. Set 60 for a smoke test.
MAX_FRAMES = None

# Exclude early TensorRT warm-up frames from steady-state summary.
WARMUP_FRAMES = 30


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
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def mean_or_zero(values):
    return round(statistics.mean(values), 3) if values else 0.0


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    return round(ordered[index], 3)


def pearson_corr(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None

    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denominator_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        return None

    return round(numerator / (denominator_x * denominator_y), 4)


def summarize_rows(rows):
    if not rows:
        return {}

    total_values = [row["total_ms"] for row in rows]
    avg_total_ms = statistics.mean(total_values)

    return {
        "frame_count": len(rows),
        "avg_raw_detection_count": mean_or_zero(
            [row["raw_detection_count"] for row in rows]
        ),
        "avg_target_detection_count": mean_or_zero(
            [row["target_detection_count"] for row in rows]
        ),
        "avg_active_track_count": mean_or_zero(
            [row["active_track_count"] for row in rows]
        ),
        "avg_inference_tracking_ms": mean_or_zero(
            [row["inference_tracking_ms"] for row in rows]
        ),
        "avg_state_build_ms": mean_or_zero(
            [row["state_build_ms"] for row in rows]
        ),
        "avg_total_ms": round(avg_total_ms, 3),
        "p95_total_ms": percentile(total_values, 0.95),
        "estimated_fps_from_avg_total_ms": (
            round(1000.0 / avg_total_ms, 3) if avg_total_ms > 0 else 0.0
        ),
        "correlation_with_active_track_count": {
            "inference_tracking_ms": pearson_corr(
                [row["active_track_count"] for row in rows],
                [row["inference_tracking_ms"] for row in rows],
            ),
            "state_build_ms": pearson_corr(
                [row["active_track_count"] for row in rows],
                [row["state_build_ms"] for row in rows],
            ),
            "total_ms": pearson_corr(
                [row["active_track_count"] for row in rows],
                [row["total_ms"] for row in rows],
            ),
        },
    }


def preload_frames(image_paths):
    print(f"[INFO] Preloading {len(image_paths)} JPG frames into RAM...")

    preload_start = time.perf_counter()
    loaded_frames = []

    for idx, img_path in enumerate(image_paths, start=1):
        img = imread_unicode(img_path)

        if img is None:
            raise RuntimeError(f"Cannot read image during preload: {img_path}")

        try:
            frame_idx = int(img_path.stem)
        except ValueError:
            frame_idx = idx

        loaded_frames.append((frame_idx, img))

        if idx % 100 == 0:
            print(f"[INFO] Preloaded {idx}/{len(image_paths)} frames")

    preload_elapsed_sec = time.perf_counter() - preload_start

    # Approximate in-memory size of decoded BGR arrays.
    decoded_bytes = sum(img.nbytes for _, img in loaded_frames)

    print(f"[INFO] Preload completed in {preload_elapsed_sec:.4f} sec")
    print(f"[INFO] Decoded frame RAM usage: {decoded_bytes / (1024 ** 2):.1f} MiB")

    return loaded_frames, preload_elapsed_sec, decoded_bytes


def convert_yolo_bytetrack_to_state():
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

    print(f"[INFO] Project root : {ROOT}")
    print(f"[INFO] Loading model: {MODEL_NAME}")

    init_start = time.perf_counter()
    model = YOLO(MODEL_NAME, task="detect")
    initialization_sec = time.perf_counter() - init_start

    print(f"[INFO] Model loaded in {initialization_sec:.3f} sec")
    print(f"[INFO] Input frames: {len(image_paths)}")
    print(f"[INFO] Warm-up frames excluded from steady summary: {WARMUP_FRAMES}")

    # Diagnostic-only: exclude storage and JPEG decode cost from the measured loop.
    loaded_frames, preload_elapsed_sec, decoded_bytes = preload_frames(image_paths)

    frames = []
    per_frame_metrics = []

    last_center_by_id = {}
    last_smoothed_velocity_by_id = {}
    velocity_alpha = 0.7

    total_detected = 0
    total_target_detected = 0
    total_tracked = 0
    skipped_no_id = 0

    process_start = time.perf_counter()

    for idx, (frame_idx, img) in enumerate(loaded_frames, start=1):
        frame_start = time.perf_counter()

        inference_tracking_start = time.perf_counter()

        results = model.track(
            source=img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        inference_tracking_elapsed = time.perf_counter() - inference_tracking_start

        state_build_start = time.perf_counter()

        objects = []
        raw_detection_count = 0
        target_detection_count = 0

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                raw_detection_count = len(boxes)

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names.get(cls_id, str(cls_id))
                    confidence = (
                        float(box.conf[0].item())
                        if box.conf is not None
                        else 0.0
                    )

                    total_detected += 1

                    if cls_name not in TARGET_CLASS_NAMES:
                        continue

                    target_detection_count += 1
                    total_target_detected += 1

                    if box.id is None:
                        skipped_no_id += 1
                        continue

                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1 = max(0.0, min(float(width), float(x1)))
                    y1 = max(0.0, min(float(height), float(y1)))
                    x2 = max(0.0, min(float(width), float(x2)))
                    y2 = max(0.0, min(float(height), float(y2)))

                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    prev_center = last_center_by_id.get(track_id)
                    prev_velocity = last_smoothed_velocity_by_id.get(track_id)

                    if prev_center is None:
                        raw_velocity = [0.0, 0.0]
                        smoothed_velocity = [0.0, 0.0]
                    else:
                        raw_velocity = [
                            center_x - prev_center[0],
                            center_y - prev_center[1],
                        ]

                        if prev_velocity is None:
                            smoothed_velocity = raw_velocity
                        else:
                            smoothed_velocity = [
                                velocity_alpha * prev_velocity[0]
                                + (1 - velocity_alpha) * raw_velocity[0],
                                velocity_alpha * prev_velocity[1]
                                + (1 - velocity_alpha) * raw_velocity[1],
                            ]

                    last_center_by_id[track_id] = [center_x, center_y]
                    last_smoothed_velocity_by_id[track_id] = smoothed_velocity

                    objects.append(
                        {
                            "id": track_id,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(confidence, 5),
                            "bbox_xywh": [
                                round(x1, 3),
                                round(y1, 3),
                                round(bbox_width, 3),
                                round(bbox_height, 3),
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
                    )

                    total_tracked += 1

        frames.append(
            {
                "frame": frame_idx,
                "timestamp": round((frame_idx - 1) / fps, 6),
                "object_count": len(objects),
                "objects": objects,
            }
        )

        state_build_elapsed = time.perf_counter() - state_build_start
        total_frame_elapsed = time.perf_counter() - frame_start

        row = {
            "frame": frame_idx,
            "raw_detection_count": raw_detection_count,
            "target_detection_count": target_detection_count,
            "active_track_count": len(objects),
            "inference_tracking_ms": round(inference_tracking_elapsed * 1000, 3),
            "state_build_ms": round(state_build_elapsed * 1000, 3),
            "total_ms": round(total_frame_elapsed * 1000, 3),
            "instant_fps": (
                round(1.0 / total_frame_elapsed, 3)
                if total_frame_elapsed > 0
                else 0.0
            ),
        }

        per_frame_metrics.append(row)

        if idx % 50 == 0:
            print(
                f"[INFO] {idx:>3}/{len(loaded_frames)} | "
                f"objects={row['active_track_count']:>2} | "
                f"infer+track={row['inference_tracking_ms']:>7.3f} ms | "
                f"state={row['state_build_ms']:>6.3f} ms | "
                f"total={row['total_ms']:>7.3f} ms | "
                f"fps={row['instant_fps']:>6.2f}"
            )

    processing_elapsed_sec = time.perf_counter() - process_start

    state = {
        "schema_version": "state_json_v1.1",
        "video_info": {
            **video_info,
            "used_frame_count": len(frames),
            "first_existing_frame": frames[0]["frame"] if frames else None,
            "last_existing_frame": frames[-1]["frame"] if frames else None,
            "source_type": "yolo_bytetrack_pred_preloaded_diagnostic",
        },
        "model_info": {
            "detector": MODEL_NAME,
            "tracker": "bytetrack.yaml",
            "conf_threshold": CONF_THRESHOLD,
            "target_class_names": sorted(list(TARGET_CLASS_NAMES)),
            "note": (
                "Diagnostic-only preload mode. JPG storage read and JPEG decode "
                "occur before the measured inference loop."
            ),
        },
        "conversion_summary": {
            "input_frames": len(image_paths),
            "used_frames": len(frames),
            "total_detected_boxes_before_class_filter": total_detected,
            "total_target_class_detections": total_target_detected,
            "total_tracked_objects_after_filter": total_tracked,
            "skipped_no_track_id": skipped_no_id,
            "initialization_sec": round(initialization_sec, 4),
            "preload_elapsed_sec": round(preload_elapsed_sec, 4),
            "decoded_frame_ram_mib": round(decoded_bytes / (1024 ** 2), 1),
            "processing_elapsed_sec": round(processing_elapsed_sec, 4),
            "estimated_processing_fps": (
                round(len(frames) / processing_elapsed_sec, 3)
                if processing_elapsed_sec > 0
                else 0.0
            ),
        },
        "frames": frames,
    }

    state_write_start = time.perf_counter()

    with open(OUT_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)

    state_write_sec = time.perf_counter() - state_write_start

    fieldnames = [
        "frame",
        "raw_detection_count",
        "target_detection_count",
        "active_track_count",
        "inference_tracking_ms",
        "state_build_ms",
        "total_ms",
        "instant_fps",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_frame_metrics)

    steady_rows = per_frame_metrics[WARMUP_FRAMES:]

    summary = {
        "experiment": "mot17_02_yolo11n_trt_fp16_preload_diagnostic",
        "diagnostic_only": True,
        "model": MODEL_NAME,
        "tracker": "bytetrack.yaml",
        "confidence_threshold": CONF_THRESHOLD,
        "input_frames": len(image_paths),
        "used_frames": len(frames),
        "warmup_frames_excluded_from_steady_summary": WARMUP_FRAMES,
        "initialization_sec": round(initialization_sec, 4),
        "preload_elapsed_sec": round(preload_elapsed_sec, 4),
        "decoded_frame_ram_mib": round(decoded_bytes / (1024 ** 2), 1),
        "processing_elapsed_sec": round(processing_elapsed_sec, 4),
        "state_json_write_sec": round(state_write_sec, 4),
        "processing_fps": (
            round(len(frames) / processing_elapsed_sec, 3)
            if processing_elapsed_sec > 0
            else 0.0
        ),
        "all_frames": summarize_rows(per_frame_metrics),
        "steady_state_after_warmup": summarize_rows(steady_rows),
        "output_files": {
            "state_json": str(OUT_PATH).replace("\\", "/"),
            "frame_timing_csv": str(CSV_PATH).replace("\\", "/"),
            "summary_json": str(SUMMARY_PATH).replace("\\", "/"),
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print()
    print(f"[OK] State JSON : {OUT_PATH}")
    print(f"[OK] Timing CSV : {CSV_PATH}")
    print(f"[OK] Summary    : {SUMMARY_PATH}")
    print(f"[INFO] Frames   : {len(frames)}")
    print(f"[INFO] Objects  : {total_tracked}")
    print(f"[INFO] Preload elapsed   : {preload_elapsed_sec:.4f} sec")
    print(f"[INFO] Decoded RAM usage : {decoded_bytes / (1024 ** 2):.1f} MiB")
    print(f"[INFO] Processing elapsed: {processing_elapsed_sec:.4f} sec")
    print(f"[INFO] Processing FPS    : {len(frames) / processing_elapsed_sec:.3f}")
    print(f"[INFO] State JSON write  : {state_write_sec:.4f} sec")

    steady_summary = summary["steady_state_after_warmup"]

    print()
    print("[STEADY STATE AFTER WARMUP]")
    print(f"- Avg total ms : {steady_summary['avg_total_ms']}")
    print(f"- P95 total ms : {steady_summary['p95_total_ms']}")
    print(f"- FPS from avg : {steady_summary['estimated_fps_from_avg_total_ms']}")

    print()
    print("[CORRELATION: ACTIVE TRACK COUNT VS LATENCY]")
    for key, value in steady_summary[
        "correlation_with_active_track_count"
    ].items():
        print(f"- {key:>22}: {value}")


if __name__ == "__main__":
    convert_yolo_bytetrack_to_state()
