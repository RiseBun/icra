"""Collect synchronized RLBench planner demonstrations from five cameras."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CAMERAS = ("front", "left_shoulder", "right_shoulder", "overhead", "wrist")
TASKS = {"open_drawer": "OpenDrawer", "insert_onto_square_peg": "InsertOntoSquarePeg"}


def camera_config(size):
    from rlbench.observation_config import CameraConfig, ObservationConfig
    def on():
        return CameraConfig(rgb=True, depth=True, point_cloud=True, mask=False,
                             image_size=(size, size), depth_in_meters=True)
    return ObservationConfig(
        front_camera=on(), left_shoulder_camera=on(), right_shoulder_camera=on(),
        overhead_camera=on(), wrist_camera=on(), joint_velocities=True,
        joint_positions=True, joint_forces=True, gripper_open=True, gripper_pose=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="open_drawer")
    ap.add_argument("--output", default="data/raw/rlbench_multiview")
    ap.add_argument("--demos", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench import tasks as task_module
    env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()),
                      obs_config=camera_config(args.image_size), headless=True)
    env.launch()
    task = env.get_task(getattr(task_module, TASKS[args.task]))
    root = Path(args.output) / args.task
    root.mkdir(parents=True, exist_ok=True)
    try:
        for demo_idx in range(args.demos):
            task.sample_variation()
            demo = task.get_demos(1, live_demos=True, max_attempts=20)[0]
            observations = list(demo)
            out = root / f"demo_{demo_idx:04d}"
            for camera in CAMERAS:
                (out / camera).mkdir(parents=True, exist_ok=True)
            low_dim, actions = [], []
            camera_points = {camera: [] for camera in CAMERAS}
            camera_depths = {camera: [] for camera in CAMERAS}
            camera_intrinsics = {}
            camera_extrinsics = {}
            for step, obs in enumerate(observations):
                low_dim.append(np.asarray(obs.get_low_dim_data(), dtype=np.float32))
                requested = obs.misc.get("joint_position_action")
                actions.append(np.asarray(requested, dtype=np.float32) if requested is not None else np.zeros(8, np.float32))
                for camera in CAMERAS:
                    rgb = np.asarray(getattr(obs, f"{camera}_rgb"), dtype=np.uint8)
                    depth = np.asarray(getattr(obs, f"{camera}_depth"), dtype=np.float32)
                    cloud = np.asarray(getattr(obs, f"{camera}_point_cloud"), dtype=np.float32)
                    from PIL import Image
                    Image.fromarray(rgb).save(out / camera / f"frame_{step:04d}.png")
                    np.save(out / camera / f"frame_{step:04d}_depth.npy", depth)
                    camera_points[camera].append(cloud)
                    camera_depths[camera].append(depth)
                    if step == 0:
                        camera_intrinsics[camera] = np.asarray(
                            obs.misc.get(f"{camera}_camera_intrinsics"), dtype=np.float32
                        )
                        camera_extrinsics[camera] = np.asarray(
                            obs.misc.get(f"{camera}_camera_extrinsics"), dtype=np.float32
                        )
                    if step == 0:
                        np.save(out / camera / "point_cloud_frame0.npy", cloud)
            low_dim_arr = np.stack(low_dim)
            actions_arr = np.stack(actions)
            for camera in CAMERAS:
                np.savez_compressed(
                    out / camera / "trajectory.npz",
                    front_point_cloud=np.stack(camera_points[camera]).astype(np.float32),
                    depth=np.stack(camera_depths[camera]).astype(np.float32),
                    low_dim=low_dim_arr,
                    actions=actions_arr,
                    success=np.asarray([1.0], np.float32),
                    collision=np.zeros((len(observations),), np.float32),
                    camera_intrinsics=camera_intrinsics[camera],
                    camera_extrinsics=camera_extrinsics[camera],
                    coordinate_frame=np.asarray("world"),
                )
            np.savez_compressed(out / "metadata.npz", low_dim=np.stack(low_dim), actions=np.stack(actions),
                                success=np.asarray([1.0], np.float32), task=np.asarray(args.task),
                                cameras=np.asarray(CAMERAS))
            print(f"saved multiview demo {out} frames={len(observations)}", flush=True)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
