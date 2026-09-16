"""Replay live RLBench demonstrations with controlled action perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from collect_rlbench_demos import TASKS


def _camera_config(size):
    from rlbench.observation_config import CameraConfig, ObservationConfig
    off = lambda: CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False)
    front = CameraConfig(
        rgb=True, depth=True, point_cloud=True, mask=False,
        image_size=(size, size), depth_in_meters=True,
    )
    return ObservationConfig(
        front_camera=front, left_shoulder_camera=off(), right_shoulder_camera=off(),
        overhead_camera=off(), wrist_camera=off(), joint_velocities=True,
        joint_positions=True, joint_forces=True, gripper_open=True, gripper_pose=True,
    )


def _expert_actions(demo):
    actions = []
    for obs in list(demo)[1:]:
        action = obs.misc.get("joint_position_action")
        if action is not None:
            actions.append(np.asarray(action, dtype=np.float32))
    if not actions:
        raise RuntimeError("live demo did not expose joint_position_action")
    return actions


def _save_rollout(observations, actions, collisions, success, root, metadata):
    root.mkdir(parents=True, exist_ok=True)
    rgb = np.stack([np.asarray(o.front_rgb, dtype=np.uint8) for o in observations])
    depth = np.stack([np.asarray(o.front_depth, dtype=np.float32) for o in observations])
    points = np.stack([np.asarray(o.front_point_cloud, dtype=np.float32) for o in observations])
    low_dim = np.stack([np.asarray(o.get_low_dim_data(), dtype=np.float32) for o in observations])
    for i, frame in enumerate(rgb):
        from PIL import Image
        Image.fromarray(frame).save(root / f"frame_{i:04d}.png")
    action_array = np.zeros((len(observations), 8), dtype=np.float32)
    action_array[:min(len(actions), len(observations))] = np.asarray(actions[:len(observations)])
    collision_array = np.zeros((len(observations),), dtype=np.float32)
    collision_array[:min(len(collisions), len(observations))] = collisions[:len(observations)]
    success_array = np.zeros((len(observations),), dtype=np.float32)
    success_array[-1] = float(success)
    first = observations[0]
    np.savez_compressed(
        root / "trajectory.npz", front_rgb=rgb, front_depth=depth,
        front_point_cloud=points, low_dim=low_dim, actions=action_array,
        success=success_array, collision=collision_array,
        front_camera_intrinsics=np.asarray(first.misc["front_camera_intrinsics"], dtype=np.float32),
        front_camera_extrinsics=np.asarray(first.misc["front_camera_extrinsics"], dtype=np.float32),
        coordinate_frame=np.asarray("world"), **metadata,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="open_drawer")
    ap.add_argument("--output", default="data/raw/rlbench_perturbed")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--noise-std", type=float, nargs="+", default=(0.01, 0.03, 0.06))
    ap.add_argument("--gripper-delay", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench import tasks as task_module

    mode = MoveArmThenGripper(JointPosition(absolute_mode=True), Discrete())
    env = Environment(mode, obs_config=_camera_config(args.image_size), headless=True)
    env.launch()
    task = env.get_task(getattr(task_module, TASKS[args.task]))
    task_root = Path(args.output) / args.task
    start_index = len(list(task_root.glob("episode_*")))
    try:
        for episode in range(args.episodes):
            task.sample_variation()
            demo = task.get_demos(1, live_demos=True, max_attempts=20)[0]
            expert = _expert_actions(demo)
            variants = [("expert", 0.0, 0)]
            variants += [(f"joint_noise_{std:g}", std, 0) for std in args.noise_std]
            variants += [("gripper_delay", 0.0, args.gripper_delay)]
            for name, noise_std, delay in variants:
                _, obs = task.reset_to_demo(demo)
                observations, executed, collisions = [obs], [], []
                succeeded = False
                for step, base in enumerate(expert):
                    action = base.copy()
                    if noise_std:
                        action[:7] += rng.normal(0.0, noise_std, 7).astype(np.float32)
                    if delay and step > 0:
                        old = expert[max(0, step - delay)][7]
                        action[7] = old
                    action[7] = float(action[7] >= 0.5)
                    try:
                        obs, reward, terminate = task.step(action)
                    except Exception:
                        break
                    executed.append(action)
                    observations.append(obs)
                    collisions.append(float(task._robot.arm.check_arm_collision()))
                    succeeded = succeeded or reward > 0.5
                    if terminate:
                        break
                destination = task_root / f"episode_{start_index + episode:04d}" / name
                _save_rollout(
                    observations, executed, collisions, succeeded, destination,
                    {"task": np.asarray(args.task), "variant": np.asarray(name),
                     "noise_std": np.asarray(noise_std, dtype=np.float32),
                     "gripper_delay": np.asarray(delay, dtype=np.int32)},
                )
                print(f"saved {destination} frames={len(observations)} success={int(succeeded)} "
                      f"collisions={int(sum(collisions))}", flush=True)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
