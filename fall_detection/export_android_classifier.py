"""Export the trained StandardScaler + logistic classifier as portable JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    model_path = Path(report["model_file"])
    if digest(model_path) != report["model_sha256"]:
        raise ValueError("Classifier hash differs from training report")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline) or len(model.steps) != 2:
        raise ValueError("Expected a StandardScaler + LogisticRegression pipeline")
    scaler, regression = model.steps[0][1], model.steps[1][1]
    if not isinstance(scaler, StandardScaler) or not isinstance(regression, LogisticRegression):
        raise ValueError("Unexpected classifier pipeline")
    if regression.coef_.shape != (1, 392) or scaler.mean_.shape != (392,):
        raise ValueError("Expected 392 temporal pose features")
    payload = {
        "schema": "fall-pose-logistic-v1",
        "feature_order": "mean,std,min,max,first,last,last-minus-first; 56 values each",
        "source_classifier_sha256": report["model_sha256"],
        "threshold": float(report["threshold_selected_on_validation"]),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": regression.coef_[0].tolist(),
        "intercept": float(regression.intercept_[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    # Verify the JSON math reproduces the saved model on random numeric vectors.
    rng = np.random.default_rng(42)
    probe = rng.normal(size=(16, 392))
    z = ((probe - payload["mean"]) / payload["scale"]) @ payload["coefficients"] + payload["intercept"]
    portable = 1 / (1 + np.exp(-z))
    if not np.allclose(portable, model.predict_proba(probe)[:, 1], atol=1e-10):
        raise RuntimeError("Portable classifier does not match the saved model")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": digest(args.output), "threshold": payload["threshold"]}, indent=2))


if __name__ == "__main__":
    main()
