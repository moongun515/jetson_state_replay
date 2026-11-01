"""Score one short, pretrimmed video with the YOLO26 pose fall baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from ultralytics import YOLO

from prepare_caucafall import extract_features
from train_temporal import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    for path in (args.video, args.pose_model, args.classifier):
        if not path.is_file():
            raise FileNotFoundError(path)

    pose = YOLO(str(args.pose_model))
    if pose.task != "pose":
        raise ValueError(f"Expected a pose checkpoint, got {pose.task}")
    classifier = joblib.load(args.classifier)
    features, found, duration = extract_features(pose, args.video, sample_frames=12, imgsz=640,
                                                batch_size=1 if args.pose_model.suffix.lower() == ".onnx" else 4)
    x = summarize(features[np.newaxis, ...], np.array([duration], dtype=np.float32))
    probability = float(classifier.predict_proba(x)[0, 1])
    print(json.dumps({
        "video": str(args.video.resolve()),
        "fall_or_fallen_probability": probability,
        "predicted_fall_or_fallen": probability >= args.threshold,
        "threshold": args.threshold,
        "pose_frames_found": found,
        "sampled_frames": 12,
        "duration_seconds": duration,
        "usage_limit": "Only short, pretrimmed clips match training; live or long video needs sliding-window validation.",
    }, indent=2))


if __name__ == "__main__":
    main()
