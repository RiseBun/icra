"""Convert RLBench trajectory.npz files into feature-contract training clips."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/rlbench_demos")
    ap.add_argument("--output", default="data/features/rlbench")
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--future", type=int, default=8)
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--split", choices=("none", "train", "val", "test"), default="none")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    args = ap.parse_args()
    files = sorted(Path(args.input).glob("**/trajectory.npz"))
    if not files:
        raise SystemExit(f"no trajectories found under {args.input}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    counter = len(list(out.glob("clip_*.npz")))
    for path in files:
        with np.load(path) as probe:
            required = {"front_point_cloud", "low_dim", "actions", "collision"}
            if not required.issubset(probe.files):
                continue
        if args.split != "none":
            group = f"{path.parent.parent.parent.name}/{path.parent.parent.name}"
            bucket = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 2**32
            if bucket < args.test_fraction:
                chosen = "test"
            elif bucket < args.test_fraction + args.val_fraction:
                chosen = "val"
            else:
                chosen = "train"
            if chosen != args.split:
                continue
        d = np.load(path)
        cloud = np.asarray(d["front_point_cloud"], dtype=np.float32)
        t, h, w, _ = cloud.shape
        if t < args.history + args.future:
            continue
        pixel = np.linspace(0, h * w - 1, args.points).astype(np.int64)
        points = cloud.reshape(t, h * w, 3)[:, pixel]
        low_dim = np.asarray(d["low_dim"], dtype=np.float32)
        actions = np.asarray(d["actions"], dtype=np.float32)
        collision = np.asarray(d["collision"], dtype=np.float32)
        demo_success = float(np.asarray(d.get("success", np.array([1.0]))).max() > 0.5)
        for start in range(0, t - args.history - args.future + 1, args.stride):
            hs = start + args.history
            fs = hs + args.future
            anchor = points[hs - 1]
            future_points = points[hs:fs][None]
            action_chunk = actions[hs - 1:fs]
            action = action_chunk.mean(axis=0, keepdims=True)
            future_disp = future_points - anchor[None, None]
            disp = np.linalg.norm(future_disp[0], axis=-1).mean(axis=0)
            affordance = (disp > np.quantile(disp, 0.75)).astype(np.float32)[None]
            coll = float(collision[hs:fs].max() > 0.5) if len(collision) else 0.0
            np.savez_compressed(
                out / f"clip_{counter:06d}.npz",
                points=points[start:hs],
                robot_state=low_dim[hs - 1],
                actions=action,
                future_points=future_points,
                affordance=affordance,
                success=np.asarray([demo_success], dtype=np.float32),
                collision=np.asarray([coll], dtype=np.float32),
                coordinate_frame=d["coordinate_frame"],
                source=np.asarray(str(path)),
            )
            counter += 1
    print(f"wrote {counter} clips to {out}")


if __name__ == "__main__":
    main()
