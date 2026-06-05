#!/usr/bin/env bash
set -euo pipefail

cd ~/jetson_state_replay
source ~/yolo_test/yolo_env/bin/activate

STATE_DIR="outputs/state_json/benchmark/confidence"
REPLAY_DIR="outputs/replay/benchmark/confidence"
METRIC_DIR="outputs/metrics/benchmark/confidence/replay"

mkdir -p "$REPLAY_DIR" "$METRIC_DIR"

for TAG in conf015 conf025; do
  STATE_PATH_VALUE=$(find "$STATE_DIR" \
    -type f \
    -name "*${TAG}*run2_state.json" \
    | head -n 1)

  if [[ -z "$STATE_PATH_VALUE" ]]; then
    echo "State JSON not found: ${TAG}"
    exit 1
  fi

  OVERLAY_PATH="$REPLAY_DIR/mot17_02_yolo11n_trt_fp16_imgsz960_${TAG}_bytetrack_overlay.mp4"
  ABSTRACT_PATH="$REPLAY_DIR/mot17_02_yolo11n_trt_fp16_imgsz960_${TAG}_bytetrack_abstract.mp4"

  echo
  echo "===== GENERATING REPLAY: ${TAG} ====="
  echo "State   : $STATE_PATH_VALUE"
  echo "Overlay : $OVERLAY_PATH"
  echo "Abstract: $ABSTRACT_PATH"

  STATE_PATH="$STATE_PATH_VALUE" \
  OVERLAY_OUT="$OVERLAY_PATH" \
  ABSTRACT_OUT="$ABSTRACT_PATH" \
    python experiments/benchmark/replay_2d_from_state_env.py \
    2>&1 | tee "$METRIC_DIR/${TAG}_replay_console.log"
done

echo
echo "===== CONFIDENCE REPLAYS COMPLETE ====="

ls -lh "$REPLAY_DIR"
