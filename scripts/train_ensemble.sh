#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: train_ensemble.sh DATA_ROOT OUTPUT_ROOT [EPOCHS]}"
OUTPUT_ROOT="${2:?usage: train_ensemble.sh DATA_ROOT OUTPUT_ROOT [EPOCHS]}"
EPOCHS="${3:-20}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniforge3/envs/vipe-cu121/bin/python}"
mkdir -p "$OUTPUT_ROOT"

pids=()
for member in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$member" "$PYTHON_BIN" scripts/train_risk4d.py \
    --data "$DATA_ROOT" --epochs "$EPOCHS" \
    --output "$OUTPUT_ROOT/member_${member}.pt" --seed "$((2027 + member))" \
    > "$OUTPUT_ROOT/member_${member}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
