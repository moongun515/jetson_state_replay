#!/usr/bin/env bash
set -euo pipefail

cd ~/jetson_state_replay
source ~/yolo_test/yolo_env/bin/activate

METRIC_OUT="outputs/metrics/benchmark/input_size"
STATE_OUT="outputs/state_json/benchmark/input_size"

BASE_METRIC_DIR="outputs/metrics/cpu_queue"
BASE_STATE_DIR="outputs/state_json/cpu_queue"

SUMMARY_SRC="$BASE_METRIC_DIR/mot17_02_yolo11n_trt_fp16_async_queue4_summary.json"
TIMING_SRC="$BASE_METRIC_DIR/mot17_02_yolo11n_trt_fp16_async_queue4_timing.csv"
STATE_SRC="$BASE_STATE_DIR/mot17_02_yolo11n_trt_fp16_async_queue4_state.json"

mkdir -p "$METRIC_OUT" "$STATE_OUT"

read_temp() {
  for f in /sys/devices/virtual/thermal/thermal_zone*/temp; do
    value=$(cat "$f" 2>/dev/null || true)

    if [[ "$value" =~ ^[0-9]+$ ]]; then
      awk -v v="$value" 'BEGIN { printf "%.3fC ", v / 1000 }'
    fi
  done

  echo
}

activate_engine() {
  local imgsz="$1"

  ln -sfn \
    "models/engines/yolo11n_trt_fp16_imgsz${imgsz}_static.engine" \
    yolo11n.engine

  echo
  echo "========================================"
  echo "ACTIVE ENGINE: imgsz=${imgsz}"
  readlink -f yolo11n.engine
  echo "Temperatures:"
  read_temp
  echo "========================================"
}

run_once() {
  local imgsz="$1"
  local run_name="$2"
  local save_state="$3"

  local prefix="mot17_02_yolo11n_trt_fp16_imgsz${imgsz}_conf015_bytetrack_queue4_maxn_super_${run_name}"
  local metric_prefix="$METRIC_OUT/$prefix"

  echo
  echo "===== imgsz=${imgsz} ${run_name} START ====="
  date
  echo "Temperatures before:"
  read_temp

  /usr/bin/tegrastats \
    --interval 1000 \
    --logfile "${metric_prefix}_tegrastats.log" &

  local tegra_pid=$!

  sleep 2

  python experiments/cpu_queue/mot17_trt_async_reader.py \
    2>&1 | tee "${metric_prefix}_console.log"

  kill "$tegra_pid" 2>/dev/null || true
  wait "$tegra_pid" 2>/dev/null || true

  cp "$SUMMARY_SRC" "${metric_prefix}_summary.json"
  cp "$TIMING_SRC" "${metric_prefix}_timing.csv"

  if [[ "$save_state" == "yes" ]]; then
    cp \
      "$STATE_SRC" \
      "$STATE_OUT/${prefix}_state.json"
  fi

  echo "Temperatures after:"
  read_temp
  echo "===== imgsz=${imgsz} ${run_name} END ====="

  sleep 3
}

echo "===== INPUT SIZE BENCHMARK START ====="
date

sudo -v
sudo nvpmodel -m 2
sudo jetson_clocks

echo
echo "===== POWER MODE ====="
sudo nvpmodel -q

# ----------------------------------------
# Warmup: 공식 평균에서는 제외
# ----------------------------------------

for SIZE in 640 960 1280; do
  activate_engine "$SIZE"
  run_once "$SIZE" "warmup" "no"
done

# ----------------------------------------
# Round 1
# ----------------------------------------

activate_engine 640
run_once 640 "run1" "yes"

activate_engine 960
run_once 960 "run1" "yes"

activate_engine 1280
run_once 1280 "run1" "yes"

# ----------------------------------------
# Round 2
# ----------------------------------------

activate_engine 1280
run_once 1280 "run2" "yes"

activate_engine 640
run_once 640 "run2" "yes"

activate_engine 960
run_once 960 "run2" "yes"

# ----------------------------------------
# Round 3
# ----------------------------------------

activate_engine 960
run_once 960 "run3" "yes"

activate_engine 1280
run_once 1280 "run3" "yes"

activate_engine 640
run_once 640 "run3" "yes"

# ----------------------------------------
# Baseline 640 복구
# ----------------------------------------

activate_engine 640

echo
echo "===== INPUT SIZE BENCHMARK FINISHED ====="
date

echo
echo "===== RESTORED BASELINE ENGINE ====="
readlink -f yolo11n.engine
