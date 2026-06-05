#!/usr/bin/env bash
set -euo pipefail

cd ~/jetson_state_replay
source ~/yolo_test/yolo_env/bin/activate

OUT_DIR="outputs/metrics/power_mode"
STATE_DIR="outputs/state_json/benchmark/power_mode"
BASE_METRIC_DIR="outputs/metrics/cpu_queue"
BASE_STATE_DIR="outputs/state_json/cpu_queue"

SUMMARY_SRC="$BASE_METRIC_DIR/mot17_02_yolo11n_trt_fp16_async_queue4_summary.json"
TIMING_SRC="$BASE_METRIC_DIR/mot17_02_yolo11n_trt_fp16_async_queue4_timing.csv"
STATE_SRC="$BASE_STATE_DIR/mot17_02_yolo11n_trt_fp16_async_queue4_state.json"

mkdir -p "$OUT_DIR" "$STATE_DIR"

read_temp() {
  for f in /sys/devices/virtual/thermal/thermal_zone*/temp; do
    value=$(cat "$f" 2>/dev/null || true)
    if [[ "$value" =~ ^[0-9]+$ ]]; then
      awk -v v="$value" 'BEGIN { printf "%.3fC ", v / 1000 }'
    fi
  done
  echo
}

apply_mode() {
  local mode_id="$1"
  local mode_name="$2"

  echo
  echo "========================================"
  echo "Applying power mode: $mode_name"
  sudo nvpmodel -m "$mode_id"
  sudo jetson_clocks
  sudo nvpmodel -q
  echo "Temperatures:"
  read_temp
  echo "========================================"
}

run_once() {
  local mode_name="$1"
  local run_name="$2"
  local prefix="$OUT_DIR/${mode_name}_${run_name}"

  echo
  echo "===== ${mode_name} ${run_name} START ====="
  date
  echo "Temperatures before:"
  read_temp

  /usr/bin/tegrastats \
    --interval 1000 \
    --logfile "${prefix}_tegrastats.log" &
  local tegra_pid=$!

  sleep 2

  python experiments/cpu_queue/mot17_trt_async_reader.py \
    2>&1 | tee "${prefix}_console.log"

  kill "$tegra_pid" 2>/dev/null || true
  wait "$tegra_pid" 2>/dev/null || true

  cp "$SUMMARY_SRC" "${prefix}_summary.json"
  cp "$TIMING_SRC" "${prefix}_timing.csv"
  cp "$STATE_SRC" "$STATE_DIR/${mode_name}_${run_name}_state.json"

  echo "Temperatures after:"
  read_temp
  echo "===== ${mode_name} ${run_name} END ====="

  sleep 3
}

echo "===== POWER MODE BENCHMARK START ====="
date

sudo -v

# Warmup: 공식 평균에서는 제외
apply_mode 1 "25w"
run_once "25w" "warmup"

apply_mode 2 "maxn_super"
run_once "maxn_super" "warmup"

# 발열 순서 편향을 줄이기 위해 번갈아 실행
for idx in 1 2 3; do
  apply_mode 1 "25w"
  run_once "25w" "run${idx}"

  apply_mode 2 "maxn_super"
  run_once "maxn_super" "run${idx}"
done

echo
echo "===== POWER MODE BENCHMARK FINISHED ====="
date
