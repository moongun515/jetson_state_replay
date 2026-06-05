#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

METRIC_DIR = ROOT / "outputs/metrics/benchmark/input_size"
STATE_DIR = ROOT / "outputs/state_json/benchmark/input_size"

GT_TOTAL_OBJECTS = 18581

TEMP_PATTERN = re.compile(r"tj@([0-9.]+)C")
POWER_PATTERN = re.compile(r"VDD_IN\s+([0-9]+)mW")
RAM_PATTERN = re.compile(r"RAM\s+([0-9]+)/")


def parse_tegrastats(path: Path) -> dict[str, float]:
    temps: list[float] = []
    powers: list[float] = []
    ram_values: list[float] = []

    for line in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        temp_match = TEMP_PATTERN.search(line)
        power_match = POWER_PATTERN.search(line)
        ram_match = RAM_PATTERN.search(line)

        if temp_match:
            temps.append(float(temp_match.group(1)))

        if power_match:
            powers.append(float(power_match.group(1)))

        if ram_match:
            ram_values.append(float(ram_match.group(1)))

    return {
        "avg_temp_c": mean(temps) if temps else float("nan"),
        "max_temp_c": max(temps) if temps else float("nan"),
        "avg_power_mw": mean(powers) if powers else float("nan"),
        "max_power_mw": max(powers) if powers else float("nan"),
        "avg_ram_mb": mean(ram_values) if ram_values else float("nan"),
    }


def count_objects(path: Path) -> int:
    data: Any = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict):
        frames = (
            data.get("frames")
            or data.get("states")
            or data.get("data")
            or []
        )
    else:
        frames = []

    total = 0

    for frame in frames:
        if isinstance(frame, dict):
            objects = frame.get("objects", [])
            total += len(objects)

    return total


def prefix(imgsz: int, run_idx: int) -> str:
    return (
        f"mot17_02_yolo11n_trt_fp16_imgsz{imgsz}"
        f"_conf015_bytetrack_queue4_maxn_super_run{run_idx}"
    )


rows: list[dict[str, float | int]] = []

for imgsz in [640, 960, 1280]:
    for run_idx in [1, 2, 3]:
        name = prefix(imgsz, run_idx)

        summary_path = METRIC_DIR / f"{name}_summary.json"
        tegra_path = METRIC_DIR / f"{name}_tegrastats.log"
        state_path = STATE_DIR / f"{name}_state.json"

        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        steady = summary["steady_state_after_warmup"]
        resources = parse_tegrastats(tegra_path)
        total_objects = count_objects(state_path)

        row = {
            "imgsz": imgsz,
            "run": run_idx,
            "pipeline_fps": float(summary["pipeline_fps"]),
            "consumer_fps": float(
                steady["consumer_fps_from_avg_ms"]
            ),
            "decode_ms": float(steady["avg_decode_ms"]),
            "inference_tracking_ms": float(
                steady["avg_inference_tracking_ms"]
            ),
            "state_build_ms": float(
                steady["avg_state_build_ms"]
            ),
            "consumer_total_ms": float(
                steady["avg_consumer_total_ms"]
            ),
            "total_objects": total_objects,
            "gt_ratio": total_objects / GT_TOTAL_OBJECTS,
            **resources,
        }

        rows.append(row)


print("===== INPUT SIZE RUN RESULTS =====")

print(
    f"{'Size':>6s} {'Run':>4s} "
    f"{'Pipe FPS':>9s} {'Cons FPS':>9s} "
    f"{'Infer+Track':>12s} {'Objects':>9s} "
    f"{'GT Ratio':>9s} {'Avg Temp':>9s} "
    f"{'Power W':>8s}"
)

for row in rows:
    print(
        f"{int(row['imgsz']):>6d} "
        f"{int(row['run']):>4d} "
        f"{row['pipeline_fps']:>9.3f} "
        f"{row['consumer_fps']:>9.3f} "
        f"{row['inference_tracking_ms']:>12.3f} "
        f"{int(row['total_objects']):>9d} "
        f"{row['gt_ratio']:>9.4f} "
        f"{row['avg_temp_c']:>9.3f} "
        f"{row['avg_power_mw'] / 1000:>8.3f}"
    )


print()
print("===== THREE-RUN AVERAGE =====")

average_rows: list[dict[str, float | int]] = []

for imgsz in [640, 960, 1280]:
    group = [row for row in rows if row["imgsz"] == imgsz]

    pipeline_values = [float(row["pipeline_fps"]) for row in group]
    consumer_values = [float(row["consumer_fps"]) for row in group]

    average_row = {
        "imgsz": imgsz,
        "pipeline_fps_avg": mean(pipeline_values),
        "pipeline_fps_sd": stdev(pipeline_values),
        "consumer_fps_avg": mean(consumer_values),
        "consumer_fps_sd": stdev(consumer_values),
        "decode_ms_avg": mean(
            float(row["decode_ms"]) for row in group
        ),
        "inference_tracking_ms_avg": mean(
            float(row["inference_tracking_ms"]) for row in group
        ),
        "state_build_ms_avg": mean(
            float(row["state_build_ms"]) for row in group
        ),
        "consumer_total_ms_avg": mean(
            float(row["consumer_total_ms"]) for row in group
        ),
        "objects_avg": mean(
            float(row["total_objects"]) for row in group
        ),
        "gt_ratio_avg": mean(
            float(row["gt_ratio"]) for row in group
        ),
        "avg_temp_c": mean(
            float(row["avg_temp_c"]) for row in group
        ),
        "max_temp_c": max(
            float(row["max_temp_c"]) for row in group
        ),
        "avg_power_w": mean(
            float(row["avg_power_mw"]) for row in group
        ) / 1000,
        "max_power_w": max(
            float(row["max_power_mw"]) for row in group
        ) / 1000,
        "avg_ram_mb": mean(
            float(row["avg_ram_mb"]) for row in group
        ),
    }

    average_rows.append(average_row)

    print()
    print(f"[imgsz={imgsz}]")
    print(
        f"Pipeline FPS       : "
        f"{average_row['pipeline_fps_avg']:.3f} "
        f"± {average_row['pipeline_fps_sd']:.3f}"
    )
    print(
        f"Consumer FPS       : "
        f"{average_row['consumer_fps_avg']:.3f} "
        f"± {average_row['consumer_fps_sd']:.3f}"
    )
    print(
        f"Decode ms          : "
        f"{average_row['decode_ms_avg']:.3f}"
    )
    print(
        f"Inference+Track ms : "
        f"{average_row['inference_tracking_ms_avg']:.3f}"
    )
    print(
        f"State build ms     : "
        f"{average_row['state_build_ms_avg']:.3f}"
    )
    print(
        f"Consumer total ms  : "
        f"{average_row['consumer_total_ms_avg']:.3f}"
    )
    print(
        f"Objects avg        : "
        f"{average_row['objects_avg']:.1f}"
    )
    print(
        f"Pred / GT ratio    : "
        f"{average_row['gt_ratio_avg']:.4f}"
    )
    print(
        f"Avg temp           : "
        f"{average_row['avg_temp_c']:.3f} C"
    )
    print(
        f"Max temp           : "
        f"{average_row['max_temp_c']:.3f} C"
    )
    print(
        f"Avg power          : "
        f"{average_row['avg_power_w']:.3f} W"
    )
    print(
        f"Max power          : "
        f"{average_row['max_power_w']:.3f} W"
    )
    print(
        f"Avg RAM            : "
        f"{average_row['avg_ram_mb']:.1f} MB"
    )


csv_path = METRIC_DIR / "input_size_benchmark_three_run_average.csv"

with csv_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(average_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(average_rows)

print()
print(f"Saved CSV: {csv_path}")
