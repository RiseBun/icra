"""Headless RLBench RGB-D capture smoke test.

Run after installing CoppeliaSim/PyRep/RLBench and setting DISPLAY, e.g.:
  COPPELIASIM_ROOT=$HOME/CoppeliaSim DISPLAY=:99 \
  python scripts/rlbench_smoke.py --task open_drawer
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("open_drawer", "insert_onto_square_peg"), default="open_drawer")
    ap.add_argument("--output", default="data/raw/rlbench_smoke")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--random-actions", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-scale", type=float, default=0.15)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.output) / args.task
    root.mkdir(parents=True, exist_ok=True)
    try:
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.tasks import InsertOntoSquarePeg, OpenDrawer
    except ImportError as exc:
        raise SystemExit(
            "RLBench/PyRep is not installed in the active environment; "
            "install them after CoppeliaSim is available."
        ) from exc

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, headless=True)
    env.launch()
    task_cls = OpenDrawer if args.task == "open_drawer" else InsertOntoSquarePeg
    task = env.get_task(task_cls)
    task.sample_variation()
    _, obs = task.reset()
    point_clouds, low_dim, actions, rewards, terminated = [], [], [], [], []
    camera_intrinsics = None
    camera_extrinsics = None
    try:
        for step in range(args.steps):
            rgb = np.asarray(obs.front_rgb)
            depth = np.asarray(obs.front_depth)
            point_clouds.append(np.asarray(obs.front_point_cloud, dtype=np.float32))
            low_dim.append(np.asarray(obs.get_low_dim_data(), dtype=np.float32))
            if camera_intrinsics is None:
                camera_intrinsics = np.asarray(obs.misc["front_camera_intrinsics"], dtype=np.float32)
                camera_extrinsics = np.asarray(obs.misc["front_camera_extrinsics"], dtype=np.float32)
            np.save(root / f"frame_{step:04d}_depth.npy", depth.astype(np.float32))
            from PIL import Image
            Image.fromarray(rgb.astype(np.uint8)).save(root / f"frame_{step:04d}.png")
            if step + 1 < args.steps:
                action = np.zeros(env.action_shape, dtype=np.float32)
                if args.random_actions:
                    action[:-1] = rng.uniform(-args.action_scale, args.action_scale, size=action.shape[0] - 1)
                    action[-1] = float(rng.integers(0, 2))
                actions.append(action)
                obs, reward, terminate = task.step(action)
                rewards.append(float(reward))
                terminated.append(bool(terminate))
                if terminate:
                    break
    finally:
        env.shutdown()
    if len(actions) < len(point_clouds):
        actions.append(np.zeros(env.action_shape, dtype=np.float32))
        rewards.append(0.0)
        terminated.append(False)
    np.savez_compressed(
        root / "trajectory.npz",
        front_point_cloud=np.stack(point_clouds),
        low_dim=np.stack(low_dim),
        actions=np.stack(actions),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=np.bool_),
        front_camera_intrinsics=camera_intrinsics,
        front_camera_extrinsics=camera_extrinsics,
        coordinate_frame=np.asarray("world"),
    )
    print(f"saved RLBench frames under {root}")


if __name__ == "__main__":
    main()
