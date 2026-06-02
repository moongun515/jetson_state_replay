import csv
import json
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# 현재는 YOLO + ByteTrack 예측 결과를 replay
STATE_PATH = ROOT / "outputs" / "state_json" / "mot17_02_gt_state.json"

REPLAY_OUT_DIR = ROOT / "outputs" / "replay"
METRICS_OUT_DIR = ROOT / "outputs" / "metrics"

OVERLAY_OUT = REPLAY_OUT_DIR / "mot17_02_gt_overlay.mp4"
ABSTRACT_OUT = REPLAY_OUT_DIR / "mot17_02_gt_abstract.mp4"

METRICS_JSON_OUT = METRICS_OUT_DIR / "mot17_02_gt_replay_metrics.json"
METRICS_CSV_OUT = METRICS_OUT_DIR / "mot17_02_gt_replay_frame_metrics.csv"
METRICS_SUMMARY_OUT = METRICS_OUT_DIR / "mot17_02_gt_replay_summary.txt"
# None이면 원본 FPS 사용.
# 확인용으로 천천히 보고 싶으면 10, 5, 2 등으로 변경 가능.
DEBUG_OUTPUT_FPS = None

# 객체 궤적을 몇 프레임까지 남길지
TRAIL_LENGTH = 45

# velocity 화살표 최대 길이 제한. 너무 길게 튀는 것 방지.
MAX_ARROW_LENGTH = 45


def imread_unicode(path: Path):
    """
    Windows 한글 경로 대응용 이미지 읽기 함수.
    cv2.imread는 한글 경로에서 실패할 수 있어서 np.fromfile + cv2.imdecode 사용.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def get_color_by_id(obj_id: int):
    """
    객체 ID별 고정 색상 생성.
    실행할 때마다 같은 id는 같은 색을 갖도록 단순 해시 사용.
    """
    rng = np.random.default_rng(seed=int(obj_id))
    color = rng.integers(80, 256, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def limit_vector(vx, vy, max_len=MAX_ARROW_LENGTH):
    length = (vx ** 2 + vy ** 2) ** 0.5

    if length == 0:
        return 0.0, 0.0

    if length <= max_len:
        return vx, vy

    scale = max_len / length
    return vx * scale, vy * scale


def draw_trajectory(frame_img, points, color):
    if len(points) < 2:
        return

    pts = list(points)

    for i in range(1, len(pts)):
        p1 = tuple(map(int, pts[i - 1]))
        p2 = tuple(map(int, pts[i]))

        # 오래된 선은 조금 얇게, 최근 선은 더 두껍게
        thickness = 1 if i < len(pts) * 0.6 else 2
        cv2.line(frame_img, p1, p2, color, thickness)


def build_score_text(obj):
    """
    GT state에는 visibility가 있고,
    YOLO + ByteTrack pred state에는 confidence가 있음.
    둘 다 없거나 None이면 score:N/A로 표시.
    """
    visibility = obj.get("visibility", None)
    confidence = obj.get("confidence", None)

    if visibility is not None:
        try:
            return f"vis:{float(visibility):.2f}"
        except (TypeError, ValueError):
            pass

    if confidence is not None:
        try:
            return f"conf:{float(confidence):.2f}"
        except (TypeError, ValueError):
            pass

    return "score:N/A"


def draw_object(frame_img, obj, trajectory_points=None):
    x1, y1, x2, y2 = map(int, obj["bbox_xyxy"])
    cx, cy = map(int, obj["center"])

    velocity = obj.get("velocity", [0.0, 0.0])
    vx, vy = velocity[0], velocity[1]

    obj_id = int(obj["id"])
    class_name = obj.get("class_name", "object")

    color = get_color_by_id(obj_id)

    if trajectory_points is not None:
        draw_trajectory(frame_img, trajectory_points, color)

    # bbox
    cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)

    # center point
    cv2.circle(frame_img, (cx, cy), 5, (0, 0, 255), -1)

    # velocity arrow
    # velocity는 프레임 간 center 차이 기반이며, 시각화를 위해 배율 적용
    arrow_vx, arrow_vy = limit_vector(float(vx) * 2.5, float(vy) * 2.5)
    end_x = int(cx + arrow_vx)
    end_y = int(cy + arrow_vy)
    cv2.arrowedLine(frame_img, (cx, cy), (end_x, end_y), (255, 255, 255), 2)

    # label
    score_text = build_score_text(obj)
    label = f"id:{obj_id} {class_name} {score_text}"

    cv2.putText(
        frame_img,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_header(frame_img, frame_idx, timestamp, object_count, mode_name, output_fps):
    text1 = f"{mode_name} | Frame: {frame_idx} | Time: {timestamp:.2f}s"
    text2 = f"Objects: {object_count} | Output FPS: {output_fps}"

    # 가독성을 위해 검은 배경 박스
    cv2.rectangle(frame_img, (20, 20), (1030, 110), (0, 0, 0), -1)

    cv2.putText(
        frame_img,
        text1,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame_img,
        text2,
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_replay():
    REPLAY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    if not STATE_PATH.exists():
        raise FileNotFoundError(f"State JSON not found: {STATE_PATH}")

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    video_info = state["video_info"]
    frames = state["frames"]

    source_fps = video_info["fps"]
    output_fps = DEBUG_OUTPUT_FPS if DEBUG_OUTPUT_FPS is not None else source_fps

    width = video_info["width"]
    height = video_info["height"]
    image_dir = Path(video_info["image_dir"])
    image_ext = video_info["image_ext"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    overlay_writer = cv2.VideoWriter(
        str(OVERLAY_OUT),
        fourcc,
        output_fps,
        (width, height),
    )

    abstract_writer = cv2.VideoWriter(
        str(ABSTRACT_OUT),
        fourcc,
        output_fps,
        (width, height),
    )

    if not overlay_writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {OVERLAY_OUT}")

    if not abstract_writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {ABSTRACT_OUT}")

    track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    written_count = 0
    skipped_count = 0
    total_objects = 0

    per_frame_metrics = []

    for frame_data in frames:
        frame_start = time.perf_counter()

        frame_idx = frame_data["frame"]
        timestamp = frame_data["timestamp"]
        objects = frame_data.get("objects", [])
        object_count = len(objects)

        img_path = image_dir / f"{frame_idx:06d}{image_ext}"

        read_start = time.perf_counter()
        original = imread_unicode(img_path)
        read_elapsed = time.perf_counter() - read_start

        if original is None:
            print(f"[WARN] Missing image or unreadable image: {img_path}")
            skipped_count += 1
            continue

        render_start = time.perf_counter()

        overlay = original.copy()
        abstract = np.zeros((height, width, 3), dtype=np.uint8)

        for obj in objects:
            obj_id = int(obj["id"])
            cx, cy = obj["center"]

            track_history[obj_id].append((cx, cy))

            draw_object(overlay, obj, track_history[obj_id])
            draw_object(abstract, obj, track_history[obj_id])

        draw_header(
            overlay,
            frame_idx,
            timestamp,
            object_count,
            "YOLO+ByteTrack Overlay Replay",
            output_fps,
        )
        draw_header(
            abstract,
            frame_idx,
            timestamp,
            object_count,
            "YOLO+ByteTrack Abstract Replay",
            output_fps,
        )

        render_elapsed = time.perf_counter() - render_start

        write_start = time.perf_counter()
        overlay_writer.write(overlay)
        abstract_writer.write(abstract)
        write_elapsed = time.perf_counter() - write_start

        frame_elapsed = time.perf_counter() - frame_start

        written_count += 1
        total_objects += object_count

        per_frame_metrics.append(
            {
                "frame": frame_idx,
                "object_count": object_count,
                "read_time_ms": round(read_elapsed * 1000, 3),
                "render_time_ms": round(render_elapsed * 1000, 3),
                "write_time_ms": round(write_elapsed * 1000, 3),
                "total_frame_time_ms": round(frame_elapsed * 1000, 3),
            }
        )

    overlay_writer.release()
    abstract_writer.release()

    total_elapsed = time.perf_counter() - total_start

    def avg(key):
        if not per_frame_metrics:
            return 0
        return sum(m[key] for m in per_frame_metrics) / len(per_frame_metrics)

    avg_read_ms = avg("read_time_ms")
    avg_render_ms = avg("render_time_ms")
    avg_write_ms = avg("write_time_ms")
    avg_frame_ms = avg("total_frame_time_ms")
    avg_objects = total_objects / written_count if written_count > 0 else 0
    processing_fps = written_count / total_elapsed if total_elapsed > 0 else 0

    metrics = {
        "state_path": str(STATE_PATH).replace("\\", "/"),
        "overlay_output": str(OVERLAY_OUT).replace("\\", "/"),
        "abstract_output": str(ABSTRACT_OUT).replace("\\", "/"),
        "source_fps": source_fps,
        "output_fps": output_fps,
        "width": width,
        "height": height,
        "input_frame_count": len(frames),
        "written_frame_count": written_count,
        "skipped_frame_count": skipped_count,
        "total_objects_drawn": total_objects,
        "avg_objects_per_frame": round(avg_objects, 3),
        "avg_read_time_ms": round(avg_read_ms, 3),
        "avg_render_time_ms": round(avg_render_ms, 3),
        "avg_write_time_ms": round(avg_write_ms, 3),
        "avg_frame_time_ms": round(avg_frame_ms, 3),
        "total_elapsed_sec": round(total_elapsed, 4),
        "estimated_processing_fps": round(processing_fps, 3),
        "trajectory": {
            "enabled": True,
            "trail_length": TRAIL_LENGTH,
        },
        "velocity_arrow": {
            "enabled": True,
            "max_arrow_length": MAX_ARROW_LENGTH,
            "note": "Arrow shows current motion direction from frame-to-frame center displacement, not AI future prediction.",
        },
        "visualization": {
            "id_color_enabled": True,
            "overlay_enabled": True,
            "abstract_enabled": True,
            "score_label": "visibility for GT, confidence for YOLO prediction.",
        },
        "per_frame_metrics": per_frame_metrics,
    }

    with open(METRICS_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(METRICS_CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "object_count",
                "read_time_ms",
                "render_time_ms",
                "write_time_ms",
                "total_frame_time_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(per_frame_metrics)

    summary_text = f"""YOLO + ByteTrack 2D Replay Metrics Summary
==========================================

State file:
{STATE_PATH}

Output files:
- Overlay : {OVERLAY_OUT}
- Abstract: {ABSTRACT_OUT}

Video info:
- Resolution: {width} x {height}
- Source FPS: {source_fps}
- Output FPS: {output_fps}
- Input frames: {len(frames)}
- Written frames: {written_count}
- Skipped frames: {skipped_count}

Object summary:
- Total objects drawn: {total_objects}
- Avg objects per frame: {avg_objects:.3f}

Performance summary:
- Total elapsed: {total_elapsed:.4f} sec
- Estimated processing FPS: {processing_fps:.3f}
- Avg read time: {avg_read_ms:.3f} ms/frame
- Avg render time: {avg_render_ms:.3f} ms/frame
- Avg write time: {avg_write_ms:.3f} ms/frame
- Avg total frame time: {avg_frame_ms:.3f} ms/frame

Visualization:
- ID-based colors: enabled
- Trajectory trail: enabled, length={TRAIL_LENGTH}
- Velocity arrow: enabled, max length={MAX_ARROW_LENGTH}
- Score label: visibility for GT, confidence for YOLO prediction
- Note: velocity arrow shows current direction based on center displacement, not future prediction.
"""

    with open(METRICS_SUMMARY_OUT, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"[INFO] Written frames: {written_count}")
    print(f"[INFO] Skipped frames: {skipped_count}")
    print(f"[INFO] Total objects drawn: {total_objects}")
    print(f"[INFO] Avg objects/frame: {avg_objects:.3f}")
    print(f"[INFO] Avg read time: {avg_read_ms:.3f} ms")
    print(f"[INFO] Avg render time: {avg_render_ms:.3f} ms")
    print(f"[INFO] Avg write time: {avg_write_ms:.3f} ms")
    print(f"[INFO] Avg frame time: {avg_frame_ms:.3f} ms")
    print(f"[INFO] Processing FPS: {processing_fps:.3f}")
    print(f"[OK] Overlay replay saved: {OVERLAY_OUT}")
    print(f"[OK] Abstract replay saved: {ABSTRACT_OUT}")
    print(f"[OK] Metrics JSON saved: {METRICS_JSON_OUT}")
    print(f"[OK] Metrics CSV saved: {METRICS_CSV_OUT}")
    print(f"[OK] Metrics summary saved: {METRICS_SUMMARY_OUT}")


if __name__ == "__main__":
    make_replay()