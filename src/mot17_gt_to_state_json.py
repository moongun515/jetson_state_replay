import json
import configparser
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]

SEQ_DIR = ROOT / "data" / "MOT17-02-DPM"
GT_PATH = SEQ_DIR / "gt" / "gt.txt"
SEQINFO_PATH = SEQ_DIR / "seqinfo.ini"
IMG_DIR = SEQ_DIR / "img1"

OUT_DIR = ROOT / "outputs" / "state_json"
OUT_PATH = OUT_DIR / "mot17_02_gt_state.json"
VELOCITY_SMOOTHING_ALPHA = 0.7


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


def get_existing_frames(img_dir: Path, image_ext: str) -> set[int]:
    frames = set()

    for img_path in img_dir.glob(f"*{image_ext}"):
        try:
            frames.add(int(img_path.stem))
        except ValueError:
            pass

    return frames


def convert_gt_to_state():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    video_info = read_seqinfo(SEQINFO_PATH)
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    image_ext = video_info["image_ext"]

    existing_frames = get_existing_frames(IMG_DIR, image_ext)

    if not existing_frames:
        raise FileNotFoundError(f"No image frames found in {IMG_DIR}")

    max_existing_frame = max(existing_frames)

    frames_dict = defaultdict(list)
    last_center_by_id = {}
    last_smoothed_velocity_by_id = {}

    with open(GT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 9:
                continue

            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            mark = float(parts[6])
            class_id = int(float(parts[7]))
            visibility = float(parts[8])

            # 현재 MVP에서는 실제 이미지가 있는 프레임만 처리
            if frame not in existing_frames:
                continue

            # MVP에서는 pedestrian/person 라벨만 사용
            if class_id != 1:
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(width), x + w)
            y2 = min(float(height), y + h)

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            prev_center = last_center_by_id.get(obj_id)
            prev_smoothed_velocity = last_smoothed_velocity_by_id.get(obj_id)

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
                    alpha = VELOCITY_SMOOTHING_ALPHA
                    smoothed_velocity = [
                        alpha * prev_smoothed_velocity[0] + (1 - alpha) * raw_velocity[0],
                        alpha * prev_smoothed_velocity[1] + (1 - alpha) * raw_velocity[1],
                ]

            last_center_by_id[obj_id] = [center_x, center_y]
            last_smoothed_velocity_by_id[obj_id] = smoothed_velocity

            obj = {
                "id": obj_id,
                "class_id": class_id,
                "class_name": "person",
                "bbox_xywh": [
                    round(x, 3),
                    round(y, 3),
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
                "visibility": visibility,
                "mark": mark,
            }

            frames_dict[frame].append(obj)

    frames = []

    for frame_idx in sorted(existing_frames):
        frames.append(
            {
                "frame": frame_idx,
                "timestamp": round((frame_idx - 1) / fps, 6),
                "objects": frames_dict.get(frame_idx, []),
            }
        )

    state = {
        "video_info": {
            **video_info,
            "used_frame_count": len(frames),
            "max_existing_frame": max_existing_frame,
            "source_type": "mot17_gt",
        },
        "frames": frames,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"[OK] State JSON saved: {OUT_PATH}")
    print(f"[INFO] Existing frames: {len(existing_frames)}")
    print(f"[INFO] Used frames: {len(frames)}")
    print(f"[INFO] Output: {OUT_PATH}")


if __name__ == "__main__":
    convert_gt_to_state()