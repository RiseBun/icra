#!/usr/bin/env bash
set -euo pipefail

ROOT="${ICRA4D_ROOT:-$HOME/icra2027_4d}"
TASK="${1:-open_drawer}"
STEPS="${2:-8}"
PY="${PYTHON_BIN:-$HOME/miniforge3/envs/vipe-cu121/bin/python}"
OMEGA_ROOT="${OMEGA_ROOT:-$HOME/VGGT-Omega}"
OMEGA_CKPT="${OMEGA_CKPT:-$OMEGA_ROOT/checkpoints/vggt_omega_1b_512.pt}"

case "$TASK" in
  open_drawer|insert_onto_square_peg) ;;
  *) echo "unsupported task: $TASK" >&2; exit 2 ;;
esac

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$HOME/CoppeliaSim}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cd "$ROOT"
RAW="data/raw/rlbench_smoke/$TASK"
OMEGA_INPUT="data/raw/rlbench_smoke/omega_input"
FEATURE="data/features/rlbench_${TASK}_omega.npz"
rm -rf "$RAW" "$OMEGA_INPUT"
mkdir -p "$OMEGA_INPUT"

xvfb-run -a -s "-screen 0 1280x1024x24" "$PY" scripts/rlbench_smoke.py \
  --task "$TASK" --steps "$STEPS" --output data/raw/rlbench_smoke
cp "$RAW"/*.png "$OMEGA_INPUT"/

"$PY" scripts/export_omega.py --repo "$OMEGA_ROOT" --checkpoint "$OMEGA_CKPT" \
  --input "$OMEGA_INPUT" --output "$FEATURE" --resolution 256 --points 1024
"$PY" scripts/feature_smoke.py "$FEATURE"
echo "pipeline complete: $FEATURE"
