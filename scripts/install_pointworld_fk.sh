#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniforge3/envs/vipe-cu121/bin/python}"

"$PYTHON_BIN" -m pip install \
  pytorch_kinematics==0.7.5 pycollada==0.6 pyrender==0.1.45
# urdfpy pins obsolete networkx==2.2. PointWorld itself uses a modern networkx,
# so install urdfpy without dependencies after the compatible stack is present.
"$PYTHON_BIN" -m pip install urdfpy==0.0.22 --no-deps
"$PYTHON_BIN" -c "import networkx, pytorch_kinematics, urdfpy; assert tuple(map(int, networkx.__version__.split('.')[:2])) >= (3, 4)"
