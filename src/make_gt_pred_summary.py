import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GT_STATE_PATH = ROOT / "outputs" / "state_json" / "mot17_02_gt_state.json"
PRED_STATE_PATH = ROOT / "outputs" / "state_json" / "mot17_02_pred_yolo_bytetrack_state.json"

GT_REPLAY_METRICS_PATH = ROOT / "outputs" / "metrics" / "mot17_02_gt_replay_metrics.json"
PRED_REPLAY_METRICS_PATH = ROOT / "outputs" / "metrics" / "mot17_02_pred_yolo_bytetrack_replay_metrics.json"
PRED_TRACK_METRICS_PATH = ROOT / "outputs" / "metrics" / "mot17_02_pred_yolo_bytetrack_metrics.json"

OUT_DIR = ROOT / "outputs" / "metrics"
OUT_JSON = OUT_DIR / "mot17_02_gt_vs_pred_summary.json"
OUT_CSV = OUT_DIR / "mot17_02_gt_vs_pred_summary.csv"
OUT_MD = OUT_DIR / "mot17_02_gt_vs_pred_summary.md"


def load_json(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_state(state: dict, label: str) -> dict:
    frames = state.get("frames", [])

    object_counts = [len(frame.get("objects", [])) for frame in frames]
    total_objects = sum(object_counts)

    ids = set()
    frames_with_objects = 0

    for frame in frames:
        objects = frame.get("objects", [])
        if objects:
            frames_with_objects += 1

        for obj in objects:
            ids.add(obj.get("id"))

    avg_objects = total_objects / len(frames) if frames else 0
    max_objects = max(object_counts) if object_counts else 0
    min_objects = min(object_counts) if object_counts else 0

    video_info = state.get("video_info", {})
    conversion_summary = state.get("conversion_summary", {})

    return {
        "label": label,
        "source_type": video_info.get("source_type"),
        "frame_count": len(frames),
        "frames_with_objects": frames_with_objects,
        "total_objects": total_objects,
        "avg_objects_per_frame": round(avg_objects, 3),
        "max_objects_per_frame": max_objects,
        "min_objects_per_frame": min_objects,
        "unique_track_ids": len(ids),
        "width": video_info.get("width"),
        "height": video_info.get("height"),
        "fps": video_info.get("fps"),
        "conversion_elapsed_sec": conversion_summary.get("elapsed_sec"),
        "estimated_processing_fps": conversion_summary.get("estimated_processing_fps"),
        "used_gt_lines": conversion_summary.get("used_gt_lines"),
        "skipped_by_class": conversion_summary.get("skipped_by_class"),
        "total_tracked_objects_after_filter": conversion_summary.get("total_tracked_objects_after_filter"),
        "total_detected_boxes_before_class_filter": conversion_summary.get("total_detected_boxes_before_class_filter"),
        "skipped_no_track_id": conversion_summary.get("skipped_no_track_id"),
    }


def summarize_replay_metrics(metrics: dict | None) -> dict:
    if not metrics:
        return {}

    return {
        "replay_written_frame_count": metrics.get("written_frame_count"),
        "replay_skipped_frame_count": metrics.get("skipped_frame_count"),
        "replay_total_objects_drawn": metrics.get("total_objects_drawn"),
        "replay_avg_objects_per_frame": metrics.get("avg_objects_per_frame"),
        "replay_avg_read_time_ms": metrics.get("avg_read_time_ms"),
        "replay_avg_render_time_ms": metrics.get("avg_render_time_ms"),
        "replay_avg_write_time_ms": metrics.get("avg_write_time_ms"),
        "replay_avg_frame_time_ms": metrics.get("avg_frame_time_ms"),
        "replay_processing_fps": metrics.get("estimated_processing_fps"),
        "replay_total_elapsed_sec": metrics.get("total_elapsed_sec"),
    }


def make_markdown(gt_summary, pred_summary, ratio_summary):
    lines = []

    lines.append("# MOT17-02 GT vs YOLO+ByteTrack Summary")
    lines.append("")
    lines.append("## 1. Experiment Overview")
    lines.append("")
    lines.append("- Dataset sequence: `MOT17-02-DPM`")
    lines.append("- Input frames: 600")
    lines.append("- Resolution: 1920 x 1080")
    lines.append("- Source FPS: 30")
    lines.append("- GT pipeline: `gt.txt → State JSON → 2D Replay`")
    lines.append("- Prediction pipeline: `MOT17 images → YOLO11 + ByteTrack → State JSON → 2D Replay`")
    lines.append("")

    lines.append("## 2. Quantitative Summary")
    lines.append("")
    lines.append("| Item | GT | YOLO+ByteTrack |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Frame count | {gt_summary['frame_count']} | {pred_summary['frame_count']} |")
    lines.append(f"| Total objects | {gt_summary['total_objects']} | {pred_summary['total_objects']} |")
    lines.append(f"| Avg objects/frame | {gt_summary['avg_objects_per_frame']} | {pred_summary['avg_objects_per_frame']} |")
    lines.append(f"| Max objects/frame | {gt_summary['max_objects_per_frame']} | {pred_summary['max_objects_per_frame']} |")
    lines.append(f"| Unique track IDs | {gt_summary['unique_track_ids']} | {pred_summary['unique_track_ids']} |")
    lines.append(f"| Conversion elapsed sec | {gt_summary.get('conversion_elapsed_sec')} | {pred_summary.get('conversion_elapsed_sec')} |")
    lines.append(f"| Estimated extraction FPS | {gt_summary.get('estimated_processing_fps')} | {pred_summary.get('estimated_processing_fps')} |")
    lines.append("")

    lines.append("## 3. Ratio Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Pred / GT total objects ratio | {ratio_summary['pred_to_gt_total_object_ratio']} |")
    lines.append(f"| Pred / GT avg objects per frame ratio | {ratio_summary['pred_to_gt_avg_object_ratio']} |")
    lines.append("")

    lines.append("## 4. Observations")
    lines.append("")
    lines.append("- GT labels are much denser and include many small or distant pedestrians.")
    lines.append("- YOLO+ByteTrack mainly detects clear, foreground, and visually confident person objects.")
    lines.append("- Distant, small, occluded, or highly overlapping people are often missed or merged.")
    lines.append("- Bicycle-riding people may still be detected as `person` by YOLO, even when MOT17 GT does not treat them as pedestrian tracking targets.")
    lines.append("- This difference is not simply a bug; it shows the difference between dataset labeling policy and general object detection behavior.")
    lines.append("")

    lines.append("## 5. Current MVP Status")
    lines.append("")
    lines.append("- MVP-0 completed: GT label based State JSON and 2D Replay.")
    lines.append("- MVP-1 completed: YOLO11 + ByteTrack based automatic State JSON and 2D Replay.")
    lines.append("- Next step: compare conditions such as confidence threshold, model size, input image size, and Jetson runtime performance.")
    lines.append("")

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_state = load_json(GT_STATE_PATH)
    pred_state = load_json(PRED_STATE_PATH)

    if gt_state is None:
        raise FileNotFoundError(f"GT state not found: {GT_STATE_PATH}")

    if pred_state is None:
        raise FileNotFoundError(f"Pred state not found: {PRED_STATE_PATH}")

    gt_summary = summarize_state(gt_state, "GT")
    pred_summary = summarize_state(pred_state, "YOLO+ByteTrack")

    gt_replay_metrics = summarize_replay_metrics(load_json(GT_REPLAY_METRICS_PATH))
    pred_replay_metrics = summarize_replay_metrics(load_json(PRED_REPLAY_METRICS_PATH))

    gt_summary.update(gt_replay_metrics)
    pred_summary.update(pred_replay_metrics)

    gt_total = gt_summary["total_objects"]
    pred_total = pred_summary["total_objects"]

    gt_avg = gt_summary["avg_objects_per_frame"]
    pred_avg = pred_summary["avg_objects_per_frame"]

    ratio_summary = {
        "pred_to_gt_total_object_ratio": round(pred_total / gt_total, 4) if gt_total else None,
        "pred_to_gt_avg_object_ratio": round(pred_avg / gt_avg, 4) if gt_avg else None,
    }

    final_summary = {
        "gt": gt_summary,
        "pred_yolo_bytetrack": pred_summary,
        "ratio_summary": ratio_summary,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    fieldnames = sorted(set(gt_summary.keys()) | set(pred_summary.keys()))
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type"] + fieldnames)
        writer.writeheader()

        writer.writerow({"type": "GT", **gt_summary})
        writer.writerow({"type": "YOLO+ByteTrack", **pred_summary})

    md = make_markdown(gt_summary, pred_summary, ratio_summary)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[OK] Summary JSON saved: {OUT_JSON}")
    print(f"[OK] Summary CSV saved: {OUT_CSV}")
    print(f"[OK] Summary Markdown saved: {OUT_MD}")
    print("")
    print("[SUMMARY]")
    print(f"GT total objects: {gt_total}")
    print(f"Pred total objects: {pred_total}")
    print(f"GT avg objects/frame: {gt_avg}")
    print(f"Pred avg objects/frame: {pred_avg}")
    print(f"Pred/GT total object ratio: {ratio_summary['pred_to_gt_total_object_ratio']}")


if __name__ == "__main__":
    main()