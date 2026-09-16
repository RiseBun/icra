"""Generate a small procedural dynamic-manipulation dataset.

This is a bridge until RLBench is installed. It contains table, target, and
occluder point clusters with action-conditioned future states and collision labels.
The file contract is identical to the eventual RLBench exporter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def box_points(center, size, count, rng):
    return center + (rng.random((count, 3), dtype=np.float32) - 0.5) * size


def make_clip(index: int, points_count: int, history: int, future: int, actions: int, seed: int):
    rng = np.random.default_rng(seed + index)
    n_target = points_count // 4
    n_occ = points_count // 4
    n_table = points_count - n_target - n_occ
    target0 = np.array([0.0, 0.0, 0.10], dtype=np.float32) + rng.normal(0, 0.03, 3)
    target_vel = rng.normal(0, 0.006, 3).astype(np.float32)
    target_vel[2] = 0
    occ = np.array([0.10, 0.0, 0.14], dtype=np.float32) + rng.normal(0, 0.02, 3)
    table = box_points(np.array([0, 0, -0.03], dtype=np.float32), np.array([1.0, 1.0, 0.05], dtype=np.float32), n_table, rng)
    history_points = []
    for t in range(history):
        target = box_points(target0 + t * target_vel, np.array([0.16, 0.12, 0.18], dtype=np.float32), n_target, rng)
        blocker = box_points(occ, np.array([0.18, 0.18, 0.24], dtype=np.float32), n_occ, rng)
        history_points.append(np.concatenate([target, blocker, table], axis=0))
    points = np.stack(history_points).astype(np.float32)

    action_values = rng.normal(0, 0.12, (actions, 10)).astype(np.float32)
    action_values[:, 2] = np.clip(action_values[:, 2], -0.04, 0.04)
    future_points = np.zeros((actions, future, points_count, 3), dtype=np.float32)
    affordance = np.zeros((actions, points_count), dtype=np.float32)
    success = np.zeros(actions, dtype=np.float32)
    collision = np.zeros(actions, dtype=np.float32)
    for a in range(actions):
        delta = action_values[a, :3]
        target_start = target0 + (history - 1) * target_vel
        target_end = target_start + delta
        hit_blocker = np.linalg.norm(target_end[:2] - occ[:2]) < 0.16 and target_end[2] < 0.30
        collision[a] = float(hit_blocker)
        success[a] = float((np.linalg.norm(delta[:2] - np.array([0.18, 0.0])) < 0.10) and not hit_blocker)
        target_mask = np.zeros(points_count, dtype=bool)
        target_mask[:n_target] = True
        affordance[a, target_mask] = float(not hit_blocker)
        for h in range(future):
            alpha = (h + 1) / future
            target = box_points(target_start + alpha * delta, np.array([0.16, 0.12, 0.18], dtype=np.float32), n_target, rng)
            blocker = box_points(occ, np.array([0.18, 0.18, 0.24], dtype=np.float32), n_occ, rng)
            future_points[a, h] = np.concatenate([target, blocker, table], axis=0)
    return {
        "points": points,
        "robot_state": np.zeros(16, dtype=np.float32),
        "actions": action_values,
        "future_points": future_points,
        "success": success,
        "collision": collision,
        "affordance": affordance,
        "coordinate_frame": np.asarray("robot_base"),
        "backbone": np.asarray("procedural"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=64)
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--output", default="data/features/synthetic")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.clips):
        np.savez_compressed(out / f"scene_{i:05d}.npz", **make_clip(i, args.points, 8, 8, 8, 2027))
    print(f"generated {args.clips} clips at {out}")


if __name__ == "__main__":
    main()
