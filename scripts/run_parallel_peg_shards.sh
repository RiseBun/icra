#!/usr/bin/env bash
set -euo pipefail

ROOT="${ICRA4D_ROOT:-$HOME/icra2027_4d}"
OUT="${PEG_EVENT_OUTPUT:-$ROOT/data/features/peg_grasp_events}"
PY="${PYTHON_BIN:-$HOME/miniforge3/envs/vipe-cu121/bin/python}"
COUNT="${PEG_SHARDS:-4}"
PER_SHARD="${PEG_EPISODES_PER_SHARD:-5}"
START="${PEG_EPISODE_START:-10}"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$HOME/CoppeliaSim}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p "$ROOT/logs" "$OUT"
pids=()
for ((shard=0; shard<COUNT; shard++)); do
  offset=$((START + shard * PER_SHARD))
  log="$ROOT/logs/peg_events_shard_${offset}.log"
  xvfb-run -a -s "-screen 0 1280x1024x24" "$PY" -u \
    "$ROOT/scripts/collect_peg_grasp_events.py" \
    --episodes "$PER_SHARD" --episode-offset "$offset" \
    --manifest-name "manifest_shard_${offset}.json" \
    --offset -2 --horizon 12 --noise-std 0.05 \
    --output "$OUT" > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
