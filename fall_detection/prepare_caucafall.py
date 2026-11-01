"""Extract YOLO26 pose sequences from CAUCAFall's OmniFall-labeled clips."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


LABEL_NAMES = (
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down",
    "lying", "stand_up", "standing", "other",
)
FALL_LABELS = {1, 2}
JOINTS = 17
FRAME_FEATURES = JOINTS * 3 + 5
SPLITS = {
    "train": {1, 2, 3, 4, 5, 6},
    "validation": {7, 8},
    "test": {9, 10},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_path(root: Path, row: dict[str, str]) -> Path:
    activity = Path(row["path"]).name
    subject = int(row["subject"])
    cam = int(row["cam"])
    label = int(row["label"])
    start, end = float(row["start"]), float(row["end"])
    if label not in range(len(LABEL_NAMES)) or subject not in range(1, 11):
        raise ValueError(f"Unexpected CAUCAFall label or subject: {row}")
    if not (0 <= start < end):
        raise ValueError(f"Invalid clip boundaries: {row}")
    filename = f"{activity}_{label}_{start:.3f}_{end:.3f}_{subject}_{cam}_caucafall.avi"
    path = root / "clips" / f"Subject.{subject}" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_frames(path: Path, sample_frames: int) -> tuple[list[np.ndarray], list[int], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count < 1 or not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Invalid video metadata: {path}: {count} frames, {fps} fps")
        indices = np.rint(np.linspace(0, count - 1, sample_frames)).astype(int).tolist()
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Frame {index} could not be decoded: {path}")
            frames.append(frame)
        return frames, indices, count / fps
    finally:
        capture.release()


def frame_features(result, frame: np.ndarray) -> tuple[np.ndarray, bool]:
    features = np.zeros(FRAME_FEATURES, dtype=np.float32)
    if result.keypoints is None or len(result.boxes) == 0:
        return features, False
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(scores * np.sqrt(np.maximum(areas, 1))))
    joints = result.keypoints.data[index].cpu().numpy()
    if joints.shape != (JOINTS, 3):
        raise ValueError(f"Expected {JOINTS} confidence-bearing COCO joints, got {joints.shape}")

    x1, y1, x2, y2 = boxes[index]
    width, height = max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    scale = max(width, height)
    joint_features = joints.astype(np.float32, copy=True)
    joint_features[:, :2] = (joint_features[:, :2] - center) / scale
    joint_features[:, 2] = np.clip(joint_features[:, 2], 0, 1)
    joint_features[joint_features[:, 2] < 0.1, :2] = 0
    frame_height, frame_width = frame.shape[:2]
    features[: JOINTS * 3] = joint_features.reshape(-1)
    features[JOINTS * 3 :] = (
        np.log(width / height),
        center[1] / frame_height,
        y2 / frame_height,
        areas[index] / (frame_width * frame_height),
        scores[index],
    )
    if not np.isfinite(features).all():
        raise ValueError("Non-finite pose features")
    return features, True


def extract_features(model: YOLO, path: Path, sample_frames: int, imgsz: int, batch_size: int = 4) -> tuple[np.ndarray, int, float]:
    frames, _, duration = read_frames(path, sample_frames)
    if batch_size == 1:
        results = [model.predict(frame, imgsz=imgsz, conf=0.15, device="cpu", verbose=False)[0]
                   for frame in frames]
    else:
        results = model.predict(frames, imgsz=imgsz, conf=0.15, batch=batch_size, device="cpu", verbose=False)
    if len(results) != len(frames):
        raise RuntimeError(f"Incomplete YOLO26 predictions for {path}")
    vectors = [frame_features(result, frame) for result, frame in zip(results, frames)]
    return np.stack([vector for vector, _ in vectors]), sum(found for _, found in vectors), duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-frames", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-check only; do not train on limited output")
    args = parser.parse_args()
    if args.sample_frames < 2 or args.imgsz < 64:
        parser.error("sample-frames must be >=2 and imgsz must be >=64")
    data_root, model_path, output = args.data_root.resolve(), args.model.resolve(), args.output.resolve()
    csv_path = data_root / "caucafall.csv"
    if not csv_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("CAUCAFall CSV or YOLO26 pose model is missing")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 258:
        raise ValueError(f"Expected 258 unique annotated CAUCAFall clips, got {len(rows)}")
    paths = [clip_path(data_root, row) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate video in CAUCAFall annotations")
    if args.limit is not None:
        rows, paths = rows[: args.limit], paths[: args.limit]

    model_hash = sha256(model_path)
    csv_hash = sha256(csv_path)
    model = YOLO(str(model_path))
    if model.task != "pose":
        raise ValueError(f"Expected a YOLO26 pose model, got {model.task}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output.parent / (output.stem + "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_features, found_frames, durations = [], [], []
    for number, (row, path) in enumerate(zip(rows, paths), 1):
        token = hashlib.sha256(
            f"{path.name}|{path.stat().st_size}|{model_hash}|{args.imgsz}|{args.sample_frames}".encode()
        ).hexdigest()[:20]
        cached = cache_dir / f"{token}.npz"
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as item:
                features = item["features"]
                found = int(item["found"])
                duration = float(item["duration"])
        else:
            features, found, duration = extract_features(model, path, args.sample_frames, args.imgsz)
            temporary = cached.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, features=features, found=found, duration=duration)
            os.replace(temporary, cached)
        if features.shape != (args.sample_frames, FRAME_FEATURES):
            raise ValueError(f"Invalid cached pose shape for {path}: {features.shape}")
        all_features.append(features)
        found_frames.append(found)
        durations.append(duration)
        if number % 25 == 0 or number == len(rows):
            print(f"Pose clips {number}/{len(rows)}", flush=True)

    subjects = np.array([int(row["subject"]) for row in rows], dtype=np.int16)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int16)
    targets = np.isin(labels, list(FALL_LABELS)).astype(np.int16)
    split = np.array([next(name for name, members in SPLITS.items() if subject in members) for subject in subjects])
    with output.open("wb") as handle:
        np.savez_compressed(
            handle, features=np.stack(all_features), labels=labels, targets=targets,
            subjects=subjects, split=split, paths=np.array([str(path) for path in paths]),
            found_frames=np.array(found_frames), durations=np.array(durations, dtype=np.float32),
        )
    manifest = {
        "status": "complete", "dataset": "CAUCAFall via OmniFall", "clips": len(rows),
        "dataset_csv_sha256": csv_hash, "model_sha256": model_hash,
        "model": str(model_path), "sample_frames": args.sample_frames, "imgsz": args.imgsz,
        "label_names": LABEL_NAMES, "positive_labels": sorted(FALL_LABELS),
        "splits_by_subject": {name: sorted(members) for name, members in SPLITS.items()},
        "split_counts": {name: int(np.sum(split == name)) for name in SPLITS},
        "pose_detection_rate": float(np.sum(found_frames) / (len(rows) * args.sample_frames)),
        "features_file": str(output), "features_sha256": sha256(output),
        "limited_smoke_output": args.limit is not None,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("clips", "split_counts", "pose_detection_rate", "features_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
