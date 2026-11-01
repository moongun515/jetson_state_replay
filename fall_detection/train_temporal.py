"""Train a small fall/fallen classifier on frozen YOLO26 pose sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(sequences: np.ndarray, durations: np.ndarray) -> np.ndarray:
    if sequences.ndim != 3 or sequences.shape[2] != 56:
        raise ValueError(f"Expected (clips, frames, 56) pose features, got {sequences.shape}")
    if len(durations) != len(sequences):
        raise ValueError("Duration count differs from clip count")
    summary = np.concatenate(
        (
            sequences.mean(axis=1), sequences.std(axis=1),
            sequences.min(axis=1), sequences.max(axis=1),
            sequences[:, 0], sequences[:, -1],
            sequences[:, -1] - sequences[:, 0],
        ),
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(summary).all():
        raise ValueError("Non-finite training features")
    return summary


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "clips": int(len(y_true)), "positive_clips": int(sum(y_true)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_alarms_per_negative_clip": float(fp / max(1, tn + fp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    features_path, output_dir = args.features.resolve(), args.output_dir.resolve()
    source_manifest = features_path.with_suffix(".manifest.json")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source["status"] != "complete" or source["limited_smoke_output"]:
        raise ValueError("Training requires the complete CAUCAFall feature set")
    if source["features_sha256"] != sha256(features_path):
        raise ValueError("Feature archive hash mismatch")
    with np.load(features_path, allow_pickle=False) as data:
        sequences = data["features"]
        durations = data["durations"]
        labels = data["labels"]
        targets = data["targets"].astype(int)
        subjects = data["subjects"]
        splits = data["split"]
        paths = data["paths"]
        found = data["found_frames"]
    if len(targets) != 258 or not np.array_equal(targets, np.isin(labels, [1, 2]).astype(int)):
        raise ValueError("Unexpected CAUCAFall clips or positive label mapping")
    unique_subjects = {name: set(subjects[splits == name]) for name in ("train", "validation", "test")}
    if any(unique_subjects[a] & unique_subjects[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Subject leakage across train/validation/test")
    if unique_subjects != {"train": {1, 2, 3, 4, 5, 6}, "validation": {7, 8}, "test": {9, 10}}:
        raise ValueError(f"Unexpected subject split: {unique_subjects}")

    x = summarize(sequences, durations)
    train, validation, test = (splits == name for name in ("train", "validation", "test"))
    if any(len(np.unique(targets[part])) != 2 for part in (train, validation, test)):
        raise ValueError("Every split must contain fall and non-fall clips")

    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        candidate = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=42),
        )
        candidate.fit(x[train], targets[train])
        val_probabilities = candidate.predict_proba(x[validation])[:, 1]
        for threshold in np.arange(0.2, 0.801, 0.05):
            score = metrics(targets[validation], val_probabilities, float(threshold))
            rank = (score["f1"], score["recall"], score["precision"],
                    -abs(float(threshold) - 0.5), -c)
            if best is None or rank > best[0]:
                best = (rank, c, float(threshold), candidate, score)
    assert best is not None
    _, c, threshold, model, validation_score = best
    test_probabilities = model.predict_proba(x[test])[:, 1]
    test_score = metrics(targets[test], test_probabilities, threshold)
    all_probabilities = model.predict_proba(x)[:, 1]
    predicted = (all_probabilities >= threshold).astype(int)

    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "fall_pose_classifier.joblib"
    joblib.dump(model, model_path)
    loaded = joblib.load(model_path)
    if not np.allclose(loaded.predict_proba(x[test])[:, 1], test_probabilities):
        raise RuntimeError("Saved classifier does not reproduce test predictions")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "subject", "split", "omnifall_label", "target_fall_or_fallen",
                         "pose_frames_found", "duration_sec", "fall_probability", "prediction"))
        for index in range(len(targets)):
            writer.writerow((paths[index], int(subjects[index]), splits[index], int(labels[index]),
                             int(targets[index]), int(found[index]), float(durations[index]),
                             float(all_probabilities[index]), int(predicted[index])))

    per_label = {}
    for label in sorted(set(int(value) for value in labels)):
        selected = test & (labels == label)
        if selected.any():
            per_label[str(label)] = {
                "name": source["label_names"][label], "clips": int(selected.sum()),
                "predicted_fall": int(predicted[selected].sum()),
            }
    report = {
        "status": "complete", "task": "fall_or_fallen_vs_other",
        "source_features": str(features_path), "source_features_sha256": sha256(features_path),
        "classifier": "StandardScaler + L2 LogisticRegression on temporal pose summaries",
        "selected_C": c, "threshold_selected_on_validation": threshold,
        "subject_splits": {key: sorted(int(item) for item in value) for key, value in unique_subjects.items()},
        "validation": validation_score, "test": test_score, "test_by_source_label": per_label,
        "model_file": str(model_path), "model_bytes": model_path.stat().st_size,
        "model_sha256": sha256(model_path), "sklearn_version": sklearn.__version__,
        "notes": [
            "YOLO26n-pose was frozen and provided COCO-17 joint features; only the small classifier was trained.",
            "CAUCAFall contains staged falls in a limited environment; test subjects are disjoint but real-world performance is unmeasured.",
            "The fall/fallen classifier cannot claim validated separation from normal lying: CAUCAFall has only one lying segment.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_C", "threshold_selected_on_validation", "validation", "test", "test_by_source_label", "model_bytes")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
