"""Collect paired local action interventions from identical RLBench histories.

Each output file is one decision point.  All candidates share the same replayed
history; only the next H actions are changed.  The nominal plan is resumed after
the intervention so terminal success and safety labels measure the causal effect
of that local action chunk rather than unrelated episode variation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policies.candidate_provider import load_candidate_provider, validate_policy_candidates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_rlbench_multiview_perturbations import (
    TASKS,
    TASK_CONTACT_SHAPES,
    _environment_collision,
    _expert_actions,
    is_task_contact_shape,
    _object_object_contacts,
)


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


def sample_grid(point_cloud: np.ndarray, count: int) -> np.ndarray:
    """Deterministically sample a fixed pixel grid from [T,H,W,3]."""
    t, h, w, _ = point_cloud.shape
    flat = point_cloud.reshape(t, h * w, 3)
    ids = np.linspace(0, h * w - 1, count, dtype=np.int64)
    return flat[:, ids].astype(np.float32)


def sample_grid_mask(mask: np.ndarray, count: int) -> np.ndarray:
    """Use exactly the same pixel indices as `sample_grid`."""
    t, h, w = mask.shape
    ids = np.linspace(0, h * w - 1, count, dtype=np.int64)
    return mask.reshape(t, h * w)[:, ids].astype(np.int64)


def contact_shape_handles(task, task_name: str):
    """Return task-required and harmful contacted shape handles at this step."""
    from pyrep.const import ObjectType

    robot_shapes = list(task._robot.arm.get_objects_in_tree(object_type=ObjectType.SHAPE))
    robot_shapes += list(task._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE))
    robot_handles = {shape.get_handle() for shape in robot_shapes}
    task_handles, harmful_handles = set(), set()
    for shape in task._scene.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE):
        if shape.get_handle() in robot_handles or not shape.is_collidable():
            continue
        if not task._robot.arm.check_arm_collision(shape):
            continue
        name = shape.get_name()
        is_task_contact = is_task_contact_shape(name, task_name)
        (task_handles if is_task_contact else harmful_handles).add(shape.get_handle())
    object_task_handles, object_harmful_handles = _object_object_contacts(task, task_name)
    task_handles.update(object_task_handles)
    harmful_handles.update(object_harmful_handles)
    return task_handles, harmful_handles


def shape_matrices(handles):
    """Read world transforms for visible segmentation handles."""
    from pyrep.objects.shape import Shape

    matrices = {}
    for handle in np.unique(handles):
        if int(handle) < 0:
            continue
        try:
            matrices[int(handle)] = np.asarray(Shape(int(handle)).get_matrix(), np.float32)
        except Exception:
            continue
    return matrices


def tracked_object_flow(reference_points, reference_handles, reference_matrices):
    """Track current visible points via their segmented object's SE(3) pose."""
    future_matrices = shape_matrices(reference_handles)
    flow = np.zeros_like(reference_points, dtype=np.float32)
    homogeneous = np.concatenate(
        (reference_points, np.ones((len(reference_points), 1), np.float32)), axis=-1)
    for handle in np.unique(reference_handles):
        handle = int(handle)
        if handle not in reference_matrices or handle not in future_matrices:
            continue
        selected = reference_handles == handle
        try:
            local = (np.linalg.inv(reference_matrices[handle]) @ homogeneous[selected].T).T
            future = (future_matrices[handle] @ local.T).T[:, :3]
        except np.linalg.LinAlgError:
            continue
        flow[selected] = future - reference_points[selected]
    return flow


def make_candidates(expert: np.ndarray, start: int, horizon: int,
                    noise_std: float, rng: np.random.Generator):
    end = min(start + horizon, len(expert))
    nominal = expert[start:end].copy()
    if len(nominal) < horizon:
        nominal = np.pad(nominal, ((0, horizon - len(nominal)), (0, 0)), mode="edge")

    # Symmetric perturbations have identical L2 magnitude, preventing a trivial
    # action-amplitude shortcut from separating the two outcomes.
    perturbation = rng.normal(0.0, noise_std, nominal[:, :7].shape).astype(np.float32)
    positive = nominal.copy(); positive[:, :7] += perturbation
    negative = nominal.copy(); negative[:, :7] -= perturbation

    delayed = nominal.copy()
    if start > 0:
        delayed[0] = expert[start - 1]
    delayed[1:] = nominal[:-1]

    gripper_delay = nominal.copy()
    gripper_delay[1:, 7] = nominal[:-1, 7]
    # Counterfactuals that target the discrete contact decision.  The position
    # trajectory remains nominal, so this tests whether the model understands
    # action-conditioned task state rather than simply action magnitude.
    gripper_hold = nominal.copy()
    gripper_hold[:, 7] = nominal[0, 7]
    gripper_flip = nominal.copy()
    gripper_flip[:, 7] = 1.0 - nominal[:, 7]
    # A stationary intervention is a matched-duration alternative that tests
    # temporal sensitivity without changing the action norm distribution too
    # aggressively.
    stationary = nominal.copy()
    if start > 0:
        stationary[:, :7] = expert[start - 1, :7]
    for candidate in (nominal, positive, negative, delayed, gripper_delay):
        candidate[:, 7] = (candidate[:, 7] >= 0.5).astype(np.float32)
    return [
        ("nominal", nominal),
        ("joint_noise_pos", positive),
        ("joint_noise_neg", negative),
        ("action_delay", delayed),
        ("gripper_delay", gripper_delay),
        ("gripper_hold", gripper_hold),
        ("gripper_flip", gripper_flip),
        ("stationary", stationary),
    ]


def policy_context(demo, start: int, history_steps: int):
    observations = list(demo)
    decision_index = min(start, len(observations) - 1)
    history = observations[max(0, decision_index - history_steps + 1):decision_index + 1]
    if len(history) < history_steps:
        history = [history[0]] * (history_steps - len(history)) + history
    return {
        "rgb_history": np.stack([observation.front_rgb for observation in history]).astype(np.uint8),
        "robot_state": np.stack([
            np.asarray(observation.get_low_dim_data(), np.float32) for observation in history
        ]),
        "joint_history": np.stack([
            np.concatenate((
                np.asarray(observation.joint_positions, np.float32),
                np.asarray([observation.gripper_open], np.float32),
            )) for observation in history
        ]),
    }


def replay_candidate(task, demo, expert: np.ndarray, start: int,
                     candidate: np.ndarray, horizon: int, task_name: str,
                     point_count: int):
    _, obs = task.reset_to_demo(demo)
    history = [obs]
    for action in expert[:start]:
        obs, _, terminate = task.step(action)
        history.append(obs)
        if terminate:
            return history, [], [], [], [], [], False, 1.0

    reference_points = sample_grid(obs.front_point_cloud[None], point_count)[0]
    reference_handles = sample_grid_mask(obs.front_mask[None], point_count)[0]
    reference_matrices = shape_matrices(reference_handles)

    future = []
    future_flow = []
    collisions = []
    rewards = []
    task_contact_handles = []
    harmful_contact_handles = []
    succeeded = False
    for action in candidate[:horizon]:
        try:
            obs, reward, terminate = task.step(action)
        except Exception:
            return (history, future, future_flow, task_contact_handles,
                    harmful_contact_handles, rewards, succeeded, 1.0)
        future.append(obs)
        future_flow.append(tracked_object_flow(
            reference_points, reference_handles, reference_matrices))
        task_handles, harmful_handles = contact_shape_handles(task, task_name)
        task_contact_handles.append(task_handles)
        harmful_contact_handles.append(harmful_handles)
        rewards.append(float(reward))
        collisions.append(float(bool(harmful_handles)))
        succeeded = succeeded or reward > 0.5
        if terminate:
            break

    # Resume the unchanged nominal plan after the local intervention.  Labels
    # therefore answer: did this candidate chunk cause eventual failure/risk?
    for action in expert[start + horizon:]:
        if succeeded:
            break
        try:
            obs, reward, terminate = task.step(action)
        except Exception:
            break
        collisions.append(_environment_collision(task, task_name))
        succeeded = succeeded or reward > 0.5
        if terminate:
            break
    return (history, future, future_flow, task_contact_handles, harmful_contact_handles,
            rewards, succeeded, float(max(collisions, default=0.0)))


def pad_observations(observations, length):
    if not observations:
        return None
    observations = list(observations[:length])
    return observations + [observations[-1]] * (length - len(observations))


def pad_sequence(sequence, length):
    if not sequence:
        return [set() for _ in range(length)]
    sequence = list(sequence[:length])
    return sequence + [sequence[-1]] * (length - len(sequence))


def pack_handle_sequences(sequences, length, max_contacts=16):
    """Pack per-step simulator shape handles without pickle/ragged arrays."""
    packed = np.full((len(sequences), length, max_contacts), -1, dtype=np.int32)
    counts = np.zeros((len(sequences), length), dtype=np.int32)
    for candidate_id, sequence in enumerate(sequences):
        for time_id, handles in enumerate(pad_sequence(sequence, length)):
            values = sorted({int(handle) for handle in handles if int(handle) >= 0})
            counts[candidate_id, time_id] = len(values)
            packed[candidate_id, time_id, :min(len(values), max_contacts)] = values[:max_contacts]
    return packed, counts


def pack_scalar_sequences(sequences, length):
    """Pack scalar simulator signals and repeat the final observed value."""
    packed = np.zeros((len(sequences), length), dtype=np.float32)
    for candidate_id, sequence in enumerate(sequences):
        values = list(sequence[:length])
        if values:
            values += [values[-1]] * (length - len(values))
            packed[candidate_id] = np.asarray(values, dtype=np.float32)
    return packed


def candidate_view_offsets(count: int) -> np.ndarray:
    """Small collision-free relative SE(3) moves around the default front view."""
    canonical = np.asarray([
        [0.12, 0.00, 0.04, 0.00, 0.12, 0.00],
        [-0.12, 0.00, 0.04, 0.00, -0.12, 0.00],
        [0.00, 0.12, 0.04, 0.12, 0.00, 0.00],
        [0.00, -0.12, 0.04, -0.12, 0.00, 0.00],
        [0.00, 0.00, 0.12, 0.00, 0.00, 0.00],
        [0.09, 0.09, 0.08, 0.08, 0.08, 0.00],
        [-0.09, 0.09, 0.08, 0.08, -0.08, 0.00],
        [0.09, -0.09, 0.08, -0.08, 0.08, 0.00],
    ], dtype=np.float32)
    if not 1 <= count <= len(canonical):
        raise ValueError(f"views must be between 1 and {len(canonical)}")
    return canonical[:count]


def capture_candidate_views(task, demo, expert, start, history_steps, point_count, view_count):
    """Capture V observations without advancing robot/task state."""
    _, observation = task.reset_to_demo(demo)
    history = [observation]
    for action in expert[:start]:
        observation, _, terminate = task.step(action)
        history.append(observation)
        if terminate:
            raise RuntimeError("task terminated before active-view decision")
    history = history[-history_steps:]
    scene = task._scene
    rgb_camera, mask_camera = scene._cam_front, scene._cam_front_mask
    rgb_position, rgb_orientation = rgb_camera.get_position(), rgb_camera.get_orientation()
    mask_position, mask_orientation = mask_camera.get_position(), mask_camera.get_orientation()
    offsets = candidate_view_offsets(view_count)
    view_rgb, view_points, view_extrinsics = [], [], []
    try:
        for offset in offsets:
            rgb_camera.set_position(np.asarray(rgb_position) + offset[:3])
            rgb_camera.set_orientation(np.asarray(rgb_orientation) + offset[3:])
            mask_camera.set_position(np.asarray(mask_position) + offset[:3])
            mask_camera.set_orientation(np.asarray(mask_orientation) + offset[3:])
            view_observation = scene.get_observation()
            rgb_sequence = [item.front_rgb for item in history[:-1]] + [view_observation.front_rgb]
            point_sequence = [item.front_point_cloud for item in history[:-1]] + [view_observation.front_point_cloud]
            view_rgb.append(np.stack(rgb_sequence).transpose(0, 3, 1, 2).astype(np.uint8))
            view_points.append(sample_grid(np.stack(point_sequence), point_count))
            base_extrinsics = [item.misc["front_camera_extrinsics"] for item in history[:-1]]
            view_extrinsics.append(np.stack(
                base_extrinsics + [view_observation.misc["front_camera_extrinsics"]]
            ).astype(np.float32))
    finally:
        rgb_camera.set_position(rgb_position)
        rgb_camera.set_orientation(rgb_orientation)
        mask_camera.set_position(mask_position)
        mask_camera.set_orientation(mask_orientation)
        scene.get_observation()
    view_cost = np.linalg.norm(offsets[:, :3], axis=-1) + 0.1 * np.linalg.norm(offsets[:, 3:], axis=-1)
    return (
        np.stack(view_rgb), np.stack(view_points).astype(np.float32),
        np.stack(view_extrinsics), offsets,
        view_cost.astype(np.float32), np.zeros(view_count, np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="open_drawer")
    ap.add_argument("--output", default="data/features/counterfactual_gt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--future", type=int, default=8)
    ap.add_argument("--action-horizon", type=int, default=None,
                    help="candidate intervention length; defaults to --future")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--critical-only", action="store_true",
                    help="sample around gripper transitions and high joint motion")
    ap.add_argument("--max-decisions", type=int, default=12)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--candidate-provider", default=None,
                    help="frozen policy sampler as module:function or file.py:function")
    ap.add_argument("--candidate-count", type=int, default=8)
    ap.add_argument("--paper-mode", action="store_true",
                    help="reject expert-perturbation debug candidates")
    args = ap.parse_args()
    if args.paper_mode and not args.candidate_provider:
        raise ValueError("paper mode requires --candidate-provider from a frozen BC/Diffusion Policy")
    provider = load_candidate_provider(args.candidate_provider) if args.candidate_provider else None
    action_horizon = args.action_horizon or args.future
    if action_horizon < args.future:
        raise ValueError("action horizon must be at least the prediction future")

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench import tasks as task_module

    rng = np.random.default_rng(args.seed)
    env = Environment(MoveArmThenGripper(JointPosition(absolute_mode=True), Discrete()),
                      obs_config=camera_config(args.image_size), headless=True)
    env.launch()
    task = env.get_task(getattr(task_module, TASKS[args.task]))
    root = Path(args.output) / args.task
    root.mkdir(parents=True, exist_ok=True)
    episode_offset = len(list(root.glob("episode_*")))
    try:
        for episode_local in range(args.episodes):
            task.sample_variation()
            demo = task.get_demos(1, live_demos=True, max_attempts=20)[0]
            expert = np.asarray(_expert_actions(demo), dtype=np.float32)
            if args.critical_only:
                speed = np.linalg.norm(np.diff(expert[:, :7], axis=0), axis=1)
                switches = np.where(np.abs(np.diff(expert[:, 7])) > 0.1)[0] + 1
                peaks = np.argsort(speed)[-max(8, len(speed) // 8):] + 1
                seeds = np.unique(np.concatenate([switches, peaks]))
                starts = sorted({int(s) for seed in seeds
                                 for s in (seed - 8, seed - 4, seed, seed + 4)
                                 if args.history - 1 <= s < len(expert) - action_horizon})
            else:
                starts = list(range(args.history - 1,
                                    max(args.history, len(expert) - action_horizon),
                                    args.stride))
            if args.max_decisions > 0:
                starts = starts[:args.max_decisions]
            episode_id = episode_offset + episode_local
            episode_root = root / f"episode_{episode_id:04d}"
            episode_root.mkdir(parents=True, exist_ok=True)
            for sample_id, start in enumerate(starts):
                if provider is None:
                    candidates = make_candidates(expert, start, action_horizon,
                                                 args.noise_std, rng)
                    candidate_source = "expert_perturbation_debug"
                else:
                    provided = provider(
                        policy_context(demo, start, args.history),
                        args.candidate_count, action_horizon,
                        args.seed + episode_id * 10000 + sample_id,
                    )
                    provided = validate_policy_candidates(
                        provided, args.candidate_count, action_horizon)
                    candidates = [
                        (f"policy_sample_{index:02d}", actions)
                        for index, actions in enumerate(provided)
                    ]
                    candidate_source = args.candidate_provider
                permutation = rng.permutation(len(candidates))
                candidates = [candidates[i] for i in permutation]
                histories, futures, flows, task_handles_all, harmful_handles_all = [], [], [], [], []
                rewards_all = []
                successes, collisions = [], []
                names_observed = []
                for name, action_chunk in candidates:
                    try:
                        (history, future, future_flow, task_handles, harmful_handles,
                         rewards, success, collision) = replay_candidate(
                            task, demo, expert, start, action_chunk, action_horizon,
                            args.task, args.points)
                    except Exception as exc:
                        print(f"candidate replay failed episode={episode_id} start={start} "
                              f"name={name}: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    padded = pad_observations(future, args.future)
                    if padded is None:
                        continue
                    histories.append(history)
                    futures.append(padded)
                    flows.append(np.stack(pad_sequence(future_flow, args.future)))
                    task_handles_all.append(pad_sequence(task_handles, args.future))
                    harmful_handles_all.append(pad_sequence(harmful_handles, args.future))
                    rewards_all.append(rewards)
                    successes.append(float(success))
                    collisions.append(float(collision))
                    names_observed.append(name)
                if len(histories) != len(candidates):
                    print(f"skip episode={episode_id} start={start}: incomplete replay", flush=True)
                    continue
                nominal_index = names_observed.index("nominal") if "nominal" in names_observed else -1
                if nominal_index >= 0 and not successes[nominal_index]:
                    print(f"episode={episode_id} start={start}: nominal replay failed; "
                          "retaining candidates for mixed-label audit", flush=True)

                try:
                    (view_rgb_history, view_points, view_camera_extrinsics, candidate_views,
                     view_cost, view_safety_cost) = capture_candidate_views(
                        task, demo, expert, start, args.history, args.points, args.views)
                except Exception as exc:
                    print(f"skip episode={episode_id} start={start}: view capture failed: {exc}", flush=True)
                    continue

                reference_history = histories[0][-args.history:]
                history_cloud = np.stack([o.front_point_cloud for o in reference_history])
                future_cloud = np.stack([
                    sample_grid(np.stack([o.front_point_cloud for o in sequence]), args.points)
                    for sequence in futures
                ])
                points = sample_grid(history_cloud, args.points)
                reference_mask = sample_grid_mask(
                    np.stack([o.front_mask for o in reference_history]), args.points)[-1]
                action_chunks = np.stack([chunk for _, chunk in candidates])
                future_point_flow = np.stack(flows).astype(np.float32)
                task_contact_map = np.zeros(future_point_flow.shape[:-1], np.float32)
                harmful_collision_map = np.zeros(future_point_flow.shape[:-1], np.float32)
                for candidate_id in range(len(candidates)):
                    for future_id in range(args.future):
                        task_contact_map[candidate_id, future_id] = np.isin(
                            reference_mask,
                            list(task_handles_all[candidate_id][future_id]),
                        )
                        harmful_collision_map[candidate_id, future_id] = np.isin(
                            reference_mask,
                            list(harmful_handles_all[candidate_id][future_id]),
                        )
                task_contact_handle_ids, task_contact_counts = pack_handle_sequences(
                    task_handles_all, args.future)
                harmful_contact_handle_ids, harmful_contact_counts = pack_handle_sequences(
                    harmful_handles_all, args.future)
                task_progress = pack_scalar_sequences(rewards_all, args.future)
                object_motion = np.linalg.norm(future_point_flow, axis=-1).mean(axis=-1)
                rgb = np.stack([o.front_rgb for o in reference_history]).transpose(0, 3, 1, 2).astype(np.uint8)
                # Keep the actual post-intervention RGB rollout for Omega-
                # ActionWorld latent supervision. Each candidate is replayed
                # from the same decision state, so this tensor is causal.
                future_rgb = np.stack([
                    np.stack([o.front_rgb for o in sequence]).transpose(0, 3, 1, 2)
                    for sequence in futures
                ]).astype(np.uint8)
                future_camera_extrinsics = np.stack([
                    np.stack([o.misc["front_camera_extrinsics"] for o in sequence])
                    for sequence in futures
                ]).astype(np.float32)
                low_dim = np.stack([
                    np.asarray(observation.get_low_dim_data(), np.float32)
                    for observation in reference_history
                ])
                current_joints = np.concatenate((
                    np.asarray(reference_history[-1].joint_positions, np.float32),
                    np.asarray([reference_history[-1].gripper_open], np.float32),
                ))
                joint_history = np.stack([
                    np.concatenate((
                        np.asarray(observation.joint_positions, np.float32),
                        np.asarray([observation.gripper_open], np.float32),
                    )) for observation in reference_history
                ])
                camera_intrinsics = np.stack([
                    observation.misc["front_camera_intrinsics"]
                    for observation in reference_history
                ]).astype(np.float32)
                camera_extrinsics = np.stack([
                    observation.misc["front_camera_extrinsics"]
                    for observation in reference_history
                ]).astype(np.float32)
                robot_base_to_world = np.asarray(task._robot.arm.get_matrix(), np.float32)
                task_embedding = np.zeros(32, np.float32)
                task_embedding[sorted(TASKS).index(args.task)] = 1.0
                output = episode_root / f"decision_{sample_id:04d}.npz"
                np.savez_compressed(
                    output, points=points, point_confidence=np.ones(points.shape[:2], np.float32),
                    point_features=np.ones((*points.shape[:2], 1), np.float32),
                    history_rgb=rgb, robot_state=low_dim, current_joints=current_joints,
                    future_rgb=future_rgb,
                    future_camera_extrinsics=future_camera_extrinsics,
                    joint_history=joint_history, camera_intrinsics=camera_intrinsics,
                    camera_extrinsics=camera_extrinsics,
                    view_camera_extrinsics=view_camera_extrinsics,
                    robot_base_to_world=robot_base_to_world,
                    candidate_actions=action_chunks,
                    actions=action_chunks.reshape(len(candidates), -1),
                    action_chunk=action_chunks.reshape(len(candidates), -1),
                    future_points=points[-1][None, None] + future_point_flow,
                    future_point_flow=future_point_flow,
                    task_contact_map=task_contact_map,
                    harmful_collision_map=harmful_collision_map,
                    task_contact_handle_ids=task_contact_handle_ids,
                    task_contact_counts=task_contact_counts,
                    harmful_contact_handle_ids=harmful_contact_handle_ids,
                    harmful_contact_counts=harmful_contact_counts,
                    task_progress=task_progress,
                    object_motion=object_motion.astype(np.float32),
                    task_embedding=task_embedding,
                    candidate_views=candidate_views, view_cost=view_cost,
                    view_safety_cost=view_safety_cost,
                    view_rgb_history=view_rgb_history,
                    view_points=view_points,
                    view_point_features=np.ones((*view_points.shape[:-1], 1), np.float32),
                    success=np.asarray(successes, np.float32),
                    collision=np.asarray(collisions, np.float32),
                    candidate_names=np.asarray(names_observed),
                    decision_step=np.asarray(start, np.int32),
                    action_horizon=np.asarray(action_horizon, np.int32),
                    future_horizon=np.asarray(args.future, np.int32),
                    task=np.asarray(args.task), coordinate_frame=np.asarray("rlbench_world_pixel_grid"),
                    candidate_source=np.asarray(candidate_source),
                    nominal_index=np.asarray(nominal_index, np.int32),
                )
                print(f"saved {output} success={successes} collision={collisions}", flush=True)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
