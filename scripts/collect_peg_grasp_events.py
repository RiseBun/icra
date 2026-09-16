"""Collect grasp-boundary counterfactuals for ``insert_onto_square_peg``.

Each retained decision point starts two actions before the first expert grasp
event and replays six matched candidate chunks from the same demonstration
state.  Labels are task outcomes (grasped/success/drop), not collision flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.peg_task_labels import derive_grasp_event_modes, has_grasp_outcome_variation


def camera_config(size: int):
    from rlbench.observation_config import CameraConfig, ObservationConfig

    on = CameraConfig(rgb=True, depth=True, point_cloud=True, mask=True,
                      image_size=(size, size), masks_as_one_channel=True,
                      depth_in_meters=True)
    off = CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False)
    return ObservationConfig(
        front_camera=on, left_shoulder_camera=off, right_shoulder_camera=off,
        overhead_camera=off, wrist_camera=off, joint_positions=True,
        joint_velocities=True, joint_forces=True, gripper_open=True,
        gripper_pose=True,
    )


def make_environment(image_size: int):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench import tasks as task_module

    env = Environment(MoveArmThenGripper(JointPosition(absolute_mode=True), Discrete()),
                      obs_config=camera_config(image_size), headless=True)
    env.launch()
    return env, env.get_task(task_module.InsertOntoSquarePeg)


def expert_actions(demo):
    actions = []
    for observation in list(demo)[1:]:
        action = observation.misc.get("joint_position_action")
        if action is not None:
            actions.append(np.asarray(action, dtype=np.float32))
    if not actions:
        raise RuntimeError("live demo did not expose joint_position_action")
    return np.stack(actions)


def find_grasp_boundary(task, demo, expert):
    _, observation = task.reset_to_demo(demo)
    previous = False
    for step, action in enumerate(expert):
        grasped = bool(task._robot.gripper.get_grasped_objects())
        if grasped and not previous:
            return step
        previous = grasped
        observation, _, terminate = task.step(action)
        if terminate:
            break
    return None


def make_candidates(expert, start, horizon, noise_std, rng, noise_l2=None):
    nominal = expert[start:start + horizon].copy()
    if len(nominal) < horizon:
        nominal = np.pad(nominal, ((0, horizon - len(nominal)), (0, 0)), mode="edge")
    perturbation = rng.normal(0.0, noise_std, nominal[:, :7].shape).astype(np.float32)
    if noise_l2 is not None:
        norms = np.linalg.norm(perturbation, axis=-1, keepdims=True)
        perturbation *= float(noise_l2) / np.maximum(norms, 1e-8)
    positive = nominal.copy(); positive[:, :7] += perturbation
    negative = nominal.copy(); negative[:, :7] -= perturbation
    flip = nominal.copy(); flip[:, 7] = 1.0 - nominal[:, 7]
    hold = nominal.copy(); hold[:, 7] = nominal[0, 7]
    delay = nominal.copy(); delay[1:, 7] = nominal[:-1, 7]
    for candidate in (nominal, positive, negative, flip, hold, delay):
        candidate[:, 7] = (candidate[:, 7] >= 0.5).astype(np.float32)
    return [("nominal", nominal), ("noise_pos", positive), ("noise_neg", negative),
            ("gripper_flip", flip), ("gripper_hold", hold), ("gripper_delay", delay)]


def replay_candidate(task, demo, expert, start, chunk, horizon):
    """Replay one candidate and resume the nominal plan to obtain success."""
    _, observation = task.reset_to_demo(demo)
    history = [observation]
    for action in expert[:start]:
        observation, _, terminate = task.step(action)
        history.append(observation)
        if terminate:
            return None
    future, grasped = [], []
    success = False
    for action in chunk[:horizon]:
        observation, reward, terminate = task.step(action)
        future.append(observation)
        grasped.append(bool(task._robot.gripper.get_grasped_objects()))
        success = success or float(reward) > 0.5
        if terminate:
            break
    for action in expert[start + horizon:]:
        if success:
            break
        observation, reward, terminate = task.step(action)
        success = success or float(reward) > 0.5
        if terminate:
            break
    if not future:
        return None
    return history, future, grasped, bool(success)


def _pad(values, length):
    values = list(values[:length])
    if not values:
        return []
    return values + [values[-1]] * (length - len(values))


def save_decision(output, rows, start, task_name, image_size, history_length, horizon,
                  noise_l2=None):
    history = rows[0]["history"][-history_length:]
    history = [history[0]] * (history_length - len(history)) + history
    history_rgb = np.stack([o.front_rgb for o in history]).transpose(0, 3, 1, 2).astype(np.uint8)
    robot_state = np.stack([np.asarray(o.get_low_dim_data(), np.float32) for o in history])
    joint_history = np.stack([
        np.concatenate((np.asarray(o.joint_positions, np.float32),
                        np.asarray([o.gripper_open], np.float32))) for o in history])
    future_rgb = np.stack([
        np.stack([o.front_rgb for o in _pad(row["future"], horizon)])
        .transpose(0, 3, 1, 2).astype(np.uint8) for row in rows])
    grasped = np.asarray([_pad(row["grasped"], horizon) for row in rows], dtype=bool)
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    grasp_failure = ~np.asarray([any(row["grasped"]) for row in rows], dtype=bool)
    contact_mode = derive_grasp_event_modes(grasped, success)
    camera_intrinsics = np.stack([
        np.asarray(o.misc["front_camera_intrinsics"], np.float32) for o in history])
    camera_extrinsics = np.stack([
        np.asarray(o.misc["front_camera_extrinsics"], np.float32) for o in history])
    actions = np.stack([row["actions"] for row in rows]).astype(np.float32)
    names = np.asarray([row["name"] for row in rows])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        history_rgb=history_rgb,
        robot_state=robot_state,
        joint_history=joint_history,
        current_joints=joint_history[-1],
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        candidate_actions=actions,
        actions=actions.reshape(len(rows), -1),
        future_rgb=future_rgb,
        grasped=grasped,
        grasped_any=~grasp_failure,
        success=success,
        task_failure=~success,
        grasp_failure=grasp_failure,
        contact_mode=contact_mode,
        contact_mode_names=np.asarray(("approach", "grasp", "drop", "completion")),
        candidate_names=names,
        decision_step=np.asarray(start, np.int32),
        future_horizon=np.asarray(horizon, np.int32),
        task=np.asarray(task_name),
        coordinate_frame=np.asarray("rlbench_world"),
        label_source=np.asarray("simulator_grasp_and_reward"),
        noise_l2=np.asarray(-1.0 if noise_l2 is None else noise_l2, np.float32),
    )
    return {
        "path": str(output), "start": int(start), "candidates": len(rows),
        "success_count": int(success.sum()), "grasp_failure_count": int(grasp_failure.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="insert_onto_square_peg")
    parser.add_argument("--output", default="data/features/peg_grasp_events")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-offset", type=int, default=0,
                        help="global episode id offset for parallel shards")
    parser.add_argument("--manifest-name", default=None,
                        help="per-shard manifest filename")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--offset", type=int, default=-2)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--noise-l2", type=float, default=None,
                        help="exact per-step L2 norm for 7-DOF noise")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.task != "insert_onto_square_peg":
        raise ValueError("grasp-event collector currently supports insert_onto_square_peg only")

    root = Path(args.output) / args.task
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for episode_local in range(args.episodes):
        episode = args.episode_offset + episode_local
        episode_rng = np.random.default_rng(args.seed + episode * 10007)
        env = None
        try:
            # A fresh simulator per episode prevents CoppeliaSim control-loop
            # state from leaking across long candidate replay batches.
            env, task = make_environment(args.image_size)
            task.sample_variation()
            demo = task.get_demos(1, live_demos=True, max_attempts=20)[0]
            expert = expert_actions(demo)
            boundary = find_grasp_boundary(task, demo, expert)
            if boundary is None:
                print(f"episode={episode}: no grasp boundary", flush=True)
                continue
            start = boundary + args.offset
            if start < args.history - 1 or start + args.horizon >= len(expert):
                print(f"episode={episode}: boundary={boundary} start={start} out of range", flush=True)
                continue
            rows = []
            for name, actions in make_candidates(expert, start, args.horizon,
                                                args.noise_std, episode_rng, args.noise_l2):
                try:
                    replay = replay_candidate(task, demo, expert, start, actions, args.horizon)
                except Exception as exc:
                    print(f"episode={episode} candidate={name} failed: {type(exc).__name__}: {exc}", flush=True)
                    replay = None
                if replay is not None:
                    history, future, grasped, success = replay
                    rows.append({"name": name, "actions": actions, "history": history,
                                 "future": future, "grasped": grasped, "success": success})
            if len(rows) != 6:
                print(f"episode={episode} start={start}: incomplete {len(rows)}/6", flush=True)
                continue
            success = np.asarray([row["success"] for row in rows])
            grasp_failure = ~np.asarray([any(row["grasped"]) for row in rows])
            if not has_grasp_outcome_variation(success, grasp_failure):
                print(f"episode={episode} start={start}: no outcome variation; filtered", flush=True)
                continue
            output = root / f"episode_{episode:04d}_start_{start:04d}.npz"
            record = save_decision(output, rows, start, args.task, args.image_size,
                                   args.history, args.horizon, args.noise_l2)
            record.update({"episode": episode, "boundary": int(boundary),
                           "offset": int(args.offset)})
            manifest.append(record)
            print(json.dumps(record), flush=True)
        except Exception as exc:
            print(f"episode={episode}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if env is not None:
                env.shutdown()
    manifest_name = args.manifest_name or f"manifest_shard_{args.episode_offset:04d}.json"
    (root / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
