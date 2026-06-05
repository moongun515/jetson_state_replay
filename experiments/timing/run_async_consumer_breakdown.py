#!/usr/bin/env python3
"""
Run the existing async-reader benchmark while measuring the internal
consumer stages without modifying the original script.

Measured stages:
- preprocess
- inference
- postprocess
- bytetrack_update
- model_track_total
- residual_framework_overhead

Default:
- CUDA synchronization enabled for clearer GPU-stage timing.
- This slightly changes the runtime behavior, so do not compare the final
  FPS directly with the original async benchmark as an official score.
"""

from __future__ import annotations

import json
import os
import runpy
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from ultralytics.engine.model import Model
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.trackers.byte_tracker import BYTETracker


ROOT = Path(__file__).resolve().parents[2]
TARGET_SCRIPT = ROOT / "experiments/cpu_queue/mot17_trt_async_reader.py"
OUTPUT_PATH = ROOT / "outputs/metrics/timing/mot17_02_async_consumer_breakdown.json"

WARMUP_CALLS = int(os.getenv("BREAKDOWN_WARMUP_CALLS", "30"))
SYNC_CUDA = os.getenv("BREAKDOWN_SYNC_CUDA", "1") == "1"

STATS: dict[str, list[float]] = {
    "preprocess_ms": [],
    "inference_ms": [],
    "postprocess_ms": [],
    "bytetrack_update_ms": [],
    "model_track_total_ms": [],
}


def cuda_sync() -> None:
    """Synchronize only when CUDA timing is explicitly enabled."""
    if SYNC_CUDA and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_call(
    stat_name: str,
    original: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    cuda_sync()
    start = time.perf_counter()
    result = original(*args, **kwargs)
    cuda_sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    STATS[stat_name].append(elapsed_ms)
    return result


# Save original methods before patching.
_original_preprocess = BasePredictor.preprocess
_original_inference = BasePredictor.inference
_original_postprocess = DetectionPredictor.postprocess
_original_bytetrack_update = BYTETracker.update
_original_model_track = Model.track


def timed_preprocess(self: BasePredictor, *args: Any, **kwargs: Any) -> Any:
    return measure_call(
        "preprocess_ms",
        lambda: _original_preprocess(self, *args, **kwargs),
    )


def timed_inference(self: BasePredictor, *args: Any, **kwargs: Any) -> Any:
    return measure_call(
        "inference_ms",
        lambda: _original_inference(self, *args, **kwargs),
    )


def timed_postprocess(
    self: DetectionPredictor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return measure_call(
        "postprocess_ms",
        lambda: _original_postprocess(self, *args, **kwargs),
    )


def timed_bytetrack_update(self: BYTETracker, *args: Any, **kwargs: Any) -> Any:
    return measure_call(
        "bytetrack_update_ms",
        lambda: _original_bytetrack_update(self, *args, **kwargs),
    )


def timed_model_track(self: Model, *args: Any, **kwargs: Any) -> Any:
    return measure_call(
        "model_track_total_ms",
        lambda: _original_model_track(self, *args, **kwargs),
    )


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return round(ordered[index], 4)


def summarize(values: list[float], warmup: int = 0) -> dict[str, Any]:
    sliced = values[warmup:] if len(values) > warmup else []
    if not sliced:
        return {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
            "total_ms": None,
        }

    return {
        "count": len(sliced),
        "avg_ms": round(statistics.mean(sliced), 4),
        "p50_ms": percentile(sliced, 0.50),
        "p95_ms": percentile(sliced, 0.95),
        "max_ms": round(max(sliced), 4),
        "total_ms": round(sum(sliced), 4),
    }


def avg(summary: dict[str, Any], key: str) -> float:
    value = summary[key]["avg_ms"]
    return float(value) if value is not None else 0.0


def main() -> None:
    if not TARGET_SCRIPT.exists():
        raise FileNotFoundError(f"Target script not found: {TARGET_SCRIPT}")

    BasePredictor.preprocess = timed_preprocess
    BasePredictor.inference = timed_inference
    DetectionPredictor.postprocess = timed_postprocess
    BYTETracker.update = timed_bytetrack_update
    Model.track = timed_model_track

    print("===== ASYNC CONSUMER BREAKDOWN =====")
    print(f"Target script        : {TARGET_SCRIPT}")
    print(f"CUDA synchronization : {SYNC_CUDA}")
    print(f"Warmup calls excluded: {WARMUP_CALLS}")
    print()
    print("Running existing async-reader benchmark...")
    print()

    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")

    steady = {
        name: summarize(values, WARMUP_CALLS)
        for name, values in STATS.items()
    }

    total_ms = avg(steady, "model_track_total_ms")
    known_ms = (
        avg(steady, "preprocess_ms")
        + avg(steady, "inference_ms")
        + avg(steady, "postprocess_ms")
        + avg(steady, "bytetrack_update_ms")
    )
    residual_ms = max(total_ms - known_ms, 0.0)

    report = {
        "target_script": str(TARGET_SCRIPT),
        "sync_cuda": SYNC_CUDA,
        "warmup_calls_excluded": WARMUP_CALLS,
        "notes": [
            "This is a diagnostic run, not the official maximum-FPS benchmark.",
            "model_track_total_ms includes Ultralytics framework and callback overhead.",
            "postprocess_ms includes bbox-oriented result processing such as NMS and Results construction.",
            "bytetrack_update_ms measures BYTETracker.update calls.",
            "residual_framework_overhead_ms is an approximate difference, not an independently timed stage.",
        ],
        "all_calls": {
            name: summarize(values, 0)
            for name, values in STATS.items()
        },
        "steady_state_after_warmup": steady,
        "derived_steady_state": {
            "known_stage_sum_ms": round(known_ms, 4),
            "residual_framework_overhead_ms": round(residual_ms, 4),
            "model_track_fps_from_avg_ms": (
                round(1000.0 / total_ms, 3) if total_ms > 0 else None
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== STEADY-STATE BREAKDOWN =====")
    for key in [
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "bytetrack_update_ms",
        "model_track_total_ms",
    ]:
        summary = steady[key]
        print(
            f"{key:28s}"
            f" avg={summary['avg_ms']} ms"
            f" | p95={summary['p95_ms']} ms"
            f" | count={summary['count']}"
        )

    print(
        f"{'residual_framework_overhead_ms':28s}"
        f" avg={report['derived_steady_state']['residual_framework_overhead_ms']} ms"
    )
    print(
        f"{'model_track_fps_from_avg_ms':28s}"
        f" {report['derived_steady_state']['model_track_fps_from_avg_ms']} FPS"
    )

    print()
    print(f"Saved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
