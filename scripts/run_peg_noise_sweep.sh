#!/usr/bin/env bash
set -euo pipefail

ROOT="${ICRA4D_ROOT:-$HOME/icra2027_4d}"
PY="${PYTHON_BIN:-$HOME/miniforge3/envs/vipe-cu121/bin/python}"
OUT_ROOT="${PEG_SWEEP_OUTPUT:-$ROOT/data/features/peg_grasp_noise_sweep}"
EPISODES="${PEG_SWEEP_EPISODES:-2}"
IMAGE_SIZE="${PEG_SWEEP_IMAGE_SIZE:-128}"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$HOME/CoppeliaSim}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs" "$OUT_ROOT"

pids=()
for spec in "002:0.02" "004:0.04" "008:0.08"; do
  tag="${spec%%:*}"
  l2="${spec##*:}"
  out="$OUT_ROOT/l2_$tag"
  log="$ROOT/logs/peg_noise_l2_$tag.log"
  xvfb-run -a -s "-screen 0 1280x1024x24" "$PY" -u \
    "$ROOT/scripts/collect_peg_grasp_events.py" \
    --episodes "$EPISODES" --episode-offset 0 \
    --manifest-name "manifest_l2_$tag.json" \
    --offset -2 --horizon 12 --noise-std 0.05 --noise-l2 "$l2" \
    --image-size "$IMAGE_SIZE" --output "$out" > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
