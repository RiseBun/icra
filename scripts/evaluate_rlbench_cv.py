"""Evaluate constant-velocity point forecasting on an RLBench trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory")
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--future", type=int, default=8)
    ap.add_argument("--points", type=int, default=1024)
    args = ap.parse_args()
    d = np.load(args.trajectory)
    cloud = np.asarray(d["front_point_cloud"], dtype=np.float32)
    t, h, w, _ = cloud.shape
    need = args.history + args.future
    if t < need:
        raise SystemExit(f"trajectory has {t} frames, need at least {need}")
    pixel = np.linspace(0, h * w - 1, args.points).astype(np.int64)
    points = cloud.reshape(t, h * w, 3)[:, pixel]
    anchor = points[args.history - 1]
    velocity = points[args.history - 1] - points[args.history - 2]
    steps = np.arange(1, args.future + 1, dtype=np.float32)[:, None, None]
    prediction = anchor[None] + steps * velocity[None]
    target = points[args.history:need]
    error = np.abs(prediction - target)
    print({
        "trajectory": str(Path(args.trajectory)),
        "coordinate_frame": d["coordinate_frame"].item(),
        "history": args.history,
        "future": args.future,
        "points": args.points,
        "constant_velocity_mae": float(error.mean()),
        "constant_velocity_rmse": float(np.sqrt((error ** 2).mean())),
        "max_step_mae": float(error.mean(axis=(1, 2)).max()),
    })


if __name__ == "__main__":
    main()
