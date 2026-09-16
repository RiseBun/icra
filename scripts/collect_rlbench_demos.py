"""Collect live RLBench demonstrations with RGB-D, point flow and robot state."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TASKS = {
    "open_drawer": "OpenDrawer",
    "insert_onto_square_peg": "InsertOntoSquarePeg",
    "open_door": "OpenDoor",
    "open_oven": "OpenOven",
    "pick_up_cup": "PickUpCup",
    "stack_blocks": "StackBlocks",
}


def _save_demo(demo, collisions, root: Path, index: int) -> None:
    observations = list(demo)
    if len(observations) < 2:
        raise RuntimeError("demo is too short")
    out = root / f"demo_{index:04d}"
    out.mkdir(parents=True, exist_ok=True)
    points, low_dim, rgbs, depths = [], [], [], []
    for step, obs in enumerate(observations):
        points.append(np.asarray(obs.front_point_cloud, dtype=np.float32))
        low_dim.append(np.asarray(obs.get_low_dim_data(), dtype=np.float32))
        rgbs.append(np.asarray(obs.front_rgb, dtype=np.uint8))
        depths.append(np.asarray(obs.front_depth, dtype=np.float32))
        from PIL import Image
        Image.fromarray(rgbs[-1]).save(out / f"frame_{step:04d}.png")
        np.save(out / f"frame_{step:04d}_depth.npy", depths[-1])
    points = np.stack(points)
    low_dim = np.stack(low_dim)
    # Live demonstrations expose the actual requested absolute joint positions.
    joint_positions = np.stack([np.asarray(o.joint_positions, dtype=np.float32) for o in observations])
    gripper = np.asarray([float(o.gripper_open) for o in observations], dtype=np.float32)[:, None]
    joint_history = np.concatenate((joint_positions, gripper), axis=1).astype(np.float32)
    actions = np.zeros((len(observations), 8), dtype=np.float32)
    actions[:-1, :7] = joint_positions[1:] - joint_positions[:-1]
    actions[:-1, 7:] = gripper[:-1]
    actions[-1, 7:] = gripper[-1]
    for step in range(1, len(observations)):
        requested = observations[step].misc.get("joint_position_action")
        if requested is not None:
            actions[step - 1] = np.asarray(requested, dtype=np.float32)
    requested_count = sum(
        observations[step].misc.get("joint_position_action") is not None
        for step in range(1, len(observations)))
    if requested_count == 0:
        raise RuntimeError(
            "live demo did not expose absolute joint_position_action; refusing to "
            "save ambiguous BC supervision")
    success = np.zeros((len(observations),), dtype=np.float32)
    success[-1] = 1.0
    collision = np.asarray(collisions[:len(observations)], dtype=np.float32)
    if len(collision) < len(observations):
        collision = np.pad(collision, (0, len(observations) - len(collision)))
    np.savez_compressed(
        out / "trajectory.npz",
        front_point_cloud=points,
        low_dim=low_dim,
        joint_history=joint_history,
        actions=actions,
        success=success,
        collision=collision,
        front_camera_intrinsics=np.asarray(observations[0].misc["front_camera_intrinsics"], dtype=np.float32),
        front_camera_extrinsics=np.asarray(observations[0].misc["front_camera_extrinsics"], dtype=np.float32),
        coordinate_frame=np.asarray("world"),
        task=np.asarray(root.name),
        action_semantics=np.asarray("absolute_joint_position"),
        requested_action_count=np.asarray(requested_count, np.int32),
    )
    print(f"saved {out} frames={len(observations)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="open_drawer")
    ap.add_argument("--output", default="data/raw/rlbench_demos")
    ap.add_argument("--demos", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()
    np.random.seed(args.seed)
    try:
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import CameraConfig, ObservationConfig
        from rlbench import tasks as task_module
    except ImportError as exc:
        raise SystemExit("RLBench/PyRep is not installed") from exc

    camera = CameraConfig(
        rgb=True, depth=True, point_cloud=True, mask=False,
        image_size=(args.image_size, args.image_size), depth_in_meters=True,
    )
    obs_config = ObservationConfig(
        front_camera=camera,
        left_shoulder_camera=CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False),
        right_shoulder_camera=CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False),
        overhead_camera=CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False),
        wrist_camera=CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False),
        joint_velocities=True, joint_positions=True, joint_forces=True,
        gripper_open=True, gripper_pose=True,
    )
    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task_cls = getattr(task_module, TASKS[args.task])
    task = env.get_task(task_cls)
    root = Path(args.output) / args.task
    root.mkdir(parents=True, exist_ok=True)
    start_index = len(list(root.glob("demo_*") ))
    try:
        for i in range(args.demos):
            collisions = []
            def callback(_obs):
                try:
                    collisions.append(float(task._robot.arm.check_arm_collision()))
                except Exception:
                    collisions.append(0.0)
            task.sample_variation()
            demos = task.get_demos(1, live_demos=True, callable_each_step=callback, max_attempts=20)
            _save_demo(demos[0], collisions, root, start_index + i)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
