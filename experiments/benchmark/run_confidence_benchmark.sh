#!/usr/bin/env bash
set -euo pipefail

cd ~/jetson_state_replay
source ~/yolo_test/yolo_env/bin/activate

METRIC_OUT="outputs/metrics/benchmark/confidence"
STATE_OUT="outputs/state_json/benchmark/confidence"

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

conf_tag() {
  case "$1" in
    0.15) echo "conf015" ;;
    0.25) echo "conf025" ;;
    *)
      echo "Unsupported confidence: $1" >&2
      exit 1
      ;;
  esac
}

activate_960_engine() {
  ln -sfn \
    models/engines/yolo11n_trt_fp16_imgsz960_static.engine \
    yolo11n.engine

  echo
  echo "===== ACTIVE ENGINE ====="
  readlink -f yolo11n.engine
}

restore_640_engine() {
  ln -sfn \
    models/engines/yolo11n_trt_fp16_imgsz640_static.engine \
    yolo11n.engine

  echo
  echo "===== RESTORED BASELINE ENGINE ====="
  readlink -f yolo11n.engine
}

run_once() {
  local conf="$1"
  local run_name="$2"
  local save_state="$3"

  local tag
  tag=$(conf_tag "$conf")

  local prefix="mot17_02_yolo11n_trt_fp16_imgsz960_${tag}_bytetrack_queue4_maxn_super_${run_name}"
  local metric_prefix="$METRIC_OUT/$prefix"

  echo
  echo "========================================"
  echo "===== ${tag} ${run_name} START ====="
  date
  echo "Temperatures before:"
  read_temp

  /usr/bin/tegrastats \
    --interval 1000 \
    --logfile "${metric_prefix}_tegrastats.log" &

  local tegra_pid=$!

  sleep 2

  CONF_THRESHOLD="$conf" \
    python experiments/benchmark/mot17_trt_async_reader_conf.py \
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
  echo "===== ${tag} ${run_name} END ====="
  echo "========================================"

  sleep 3
}

echo "===== CONFIDENCE BENCHMARK START ====="
date

sudo -v
sudo nvpmodel -m 2
sudo jetson_clocks

echo
echo "===== POWER MODE ====="
sudo nvpmodel -q

activate_960_engine

# ----------------------------------------
# Warmup: 공식 평균에서 제외
# ----------------------------------------

run_once 0.15 "warmup" "no"
run_once 0.25 "warmup" "no"

# ----------------------------------------
# Round 1
# ----------------------------------------

run_once 0.15 "run1" "yes"
run_once 0.25 "run1" "yes"

# ----------------------------------------
# Round 2: 순서 반전
# ----------------------------------------

run_once 0.25 "run2" "yes"
run_once 0.15 "run2" "yes"

# ----------------------------------------
# Round 3
# ----------------------------------------

run_once 0.15 "run3" "yes"
run_once 0.25 "run3" "yes"

restore_640_engine

echo
echo "===== CONFIDENCE BENCHMARK FINISHED ====="
date
