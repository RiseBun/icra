"""Compare zero-motion and constant-velocity forecasts on feature clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    args = ap.parse_args()
    files = sorted(Path(args.data).glob("**/*.npz"))
    zero, cv = [], []
    for path in files:
        d = np.load(path)
        points = np.asarray(d["points"], dtype=np.float32)
        target = np.asarray(d["future_points"], dtype=np.float32)[0]
        anchor = points[-1]
        zero_pred = np.broadcast_to(anchor, target.shape)
        velocity = points[-1] - points[-2]
        steps = np.arange(1, len(target) + 1, dtype=np.float32)[:, None, None]
        cv_pred = anchor[None] + steps * velocity[None]
        zero.append(np.abs(zero_pred - target).mean())
        cv.append(np.abs(cv_pred - target).mean())
    print({
        "clips": len(files),
        "zero_motion_mae": float(np.mean(zero)),
        "constant_velocity_mae": float(np.mean(cv)),
        "constant_velocity_improvement_pct": float(100.0 * (np.mean(zero) - np.mean(cv)) / max(np.mean(zero), 1e-8)),
    })


if __name__ == "__main__":
    main()
