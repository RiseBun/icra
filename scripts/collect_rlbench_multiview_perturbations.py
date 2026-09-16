"""Replay RLBench demonstrations while recording synchronized multi-view perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from collect_rlbench_demos import TASKS


CAMERAS = ("front", "left_shoulder", "right_shoulder", "overhead", "wrist")


def _camera_config(size, cameras=CAMERAS):
    from rlbench.observation_config import CameraConfig, ObservationConfig

    def on():
        return CameraConfig(rgb=True, depth=True, point_cloud=True, mask=False,
                            image_size=(size, size), depth_in_meters=True)

    def off():
        return CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False)

    cameras = set(cameras)

    return ObservationConfig(
        front_camera=on() if "front" in cameras else off(),
        left_shoulder_camera=on() if "left_shoulder" in cameras else off(),
        right_shoulder_camera=on() if "right_shoulder" in cameras else off(),
        overhead_camera=on() if "overhead" in cameras else off(),
        wrist_camera=on() if "wrist" in cameras else off(), joint_velocities=True,
        joint_positions=True, joint_forces=True, gripper_open=True, gripper_pose=True,
    )


def _expert_actions(demo):
    actions = []
    # RLBench records the executed waypoint action in the observation produced
    # after that action.  The initial observation intentionally has no action.
    for obs in list(demo)[1:]:
        action = obs.misc.get("joint_position_action")
        if action is not None:
            actions.append(np.asarray(action, dtype=np.float32))
    if not actions:
        raise RuntimeError("live demo did not expose joint_position_action")
    return actions


TASK_CONTACT_SHAPES = {
    # Opening the drawer necessarily contacts the drawer body; count only
    # contacts with other scene geometry as safety collisions.
    "open_drawer": {"drawer_top", "drawer_middle", "drawer_bottom"},
    # The ring is grasped and inserted onto one of the colored pillars; these
    # contacts are task-required and must not be counted as safety collisions.
    "insert_onto_square_peg": {"square_ring", "pillar0", "pillar1", "pillar2"},
}


def is_task_contact_shape(shape_name: str, task_name: str) -> bool:
    """Match task-contact geometry despite RLBench visual/respondable suffixes."""
    normalized = str(shape_name).lower()
    tokens = {str(value).lower() for value in TASK_CONTACT_SHAPES.get(task_name, set())}
    if any(token in normalized for token in tokens):
        return True
    return task_name == "open_drawer" and (
        "drawer" in normalized or "handle" in normalized
    )


def _named_shape(name):
    """Return a named shape when it exists, without making scene replay brittle."""
    from pyrep.objects.shape import Shape
    try:
        return Shape(name)
    except Exception:
        return None


def _target_pillar(task):
    """Infer the selected peg from the task success-centre position."""
    try:
        centre = np.asarray(task._success_centre.get_position(), dtype=np.float32)
    except Exception:
        return None
    pillars = [shape for shape in (_named_shape(f"pillar{i}") for i in range(3))
               if shape is not None]
    if not pillars:
        return None
    return min(pillars, key=lambda shape: float(
        np.linalg.norm(np.asarray(shape.get_position(), dtype=np.float32)[:2] - centre[:2])))


def _object_object_contacts(task, task_name):
    """Detect held-object contacts that arm-vs-shape checks cannot see."""
    if task_name != "insert_onto_square_peg":
        return set(), set()
    from pyrep.const import ObjectType
    try:
        grasped = list(task._robot.gripper.get_grasped_objects())
    except Exception:
        grasped = []
    target = _target_pillar(task)
    cache = getattr(task, "_icra_contact_shape_cache", None)
    if cache is None:
        try:
            scene_shapes = list(task._scene.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE))
        except Exception:
            scene_shapes = []
        # A grasped object can be re-parented below the gripper and disappear
        # from the scene-root traversal; explicitly add named task shapes.
        seen = {shape.get_handle() for shape in scene_shapes}
        for shape in (_named_shape(f"pillar{i}") for i in range(3)):
            if shape is not None and shape.get_handle() not in seen:
                scene_shapes.append(shape)
                seen.add(shape.get_handle())
        robot_handles = {shape.get_handle() for shape in
                         task._robot.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)}
        robot_handles.update(shape.get_handle() for shape in
                             task._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE))
        cache = (scene_shapes, robot_handles, target_handle)
        setattr(task, "_icra_contact_shape_cache", cache)
    scene_shapes, robot_handles, target_handle = cache
    task_handles, harmful_handles = set(), set()
    target_handle = target.get_handle() if target is not None else None
    for obj in grasped:
        for shape in scene_shapes:
            if shape.get_handle() in robot_handles or obj.get_handle() == shape.get_handle():
                continue
            try:
                if not shape.is_collidable():
                    continue
            except Exception:
                continue
            try:
                colliding = bool(obj.check_collision(shape))
            except Exception:
                colliding = False
            if not colliding:
                continue
            # Only the selected pillar is required for insertion. Contacts
            # with workspace/table/distractor pillars are harmful outcomes.
            if shape.get_handle() == target_handle:
                task_handles.add(shape.get_handle())
            else:
                harmful_handles.add(shape.get_handle())
    return task_handles, harmful_handles


def _environment_collision(task, task_name):
    """Exclude robot self-collision and task-required contact shapes."""
    from pyrep.const import ObjectType
    robot_shapes = list(task._robot.arm.get_objects_in_tree(object_type=ObjectType.SHAPE))
    robot_shapes += list(task._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE))
    for shape in task._scene.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE):
        if any(shape.get_handle() == robot.get_handle() for robot in robot_shapes) or not shape.is_collidable():
            continue
        shape_name = shape.get_name()
        if is_task_contact_shape(shape_name, task_name):
            continue
        if task._robot.arm.check_arm_collision(shape):
            return 1.0
    _, harmful_handles = _object_object_contacts(task, task_name)
    return float(bool(harmful_handles))


def _save_rollout(observations, actions, collisions, success, root, metadata,
                  cameras=CAMERAS):
    root.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    low_dim = np.stack([np.asarray(o.get_low_dim_data(), dtype=np.float32) for o in observations])
    action_array = np.zeros((len(observations), 8), dtype=np.float32)
    if actions:
        action_array[:min(len(actions), len(observations))] = np.asarray(actions[:len(observations)])
    collision_array = np.zeros((len(observations),), dtype=np.float32)
    collision_array[:min(len(collisions), len(observations))] = collisions[:len(observations)]
    success_array = np.zeros((len(observations),), dtype=np.float32)
    success_array[-1] = float(success)

    for camera in cameras:
        camera_root = root / camera
        camera_root.mkdir(parents=True, exist_ok=True)
        rgb = np.stack([np.asarray(getattr(o, f"{camera}_rgb"), dtype=np.uint8) for o in observations])
        depth = np.stack([np.asarray(getattr(o, f"{camera}_depth"), dtype=np.float32) for o in observations])
        points = np.stack([np.asarray(getattr(o, f"{camera}_point_cloud"), dtype=np.float32) for o in observations])
        for i, frame in enumerate(rgb):
            Image.fromarray(frame).save(camera_root / f"frame_{i:04d}.png")
            np.save(camera_root / f"frame_{i:04d}_depth.npy", depth[i])
        first = observations[0]
        intrinsics = np.asarray(first.misc[f"{camera}_camera_intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(first.misc[f"{camera}_camera_extrinsics"], dtype=np.float32)
        np.savez_compressed(
            camera_root / "trajectory.npz", front_rgb=rgb, front_depth=depth,
            front_point_cloud=points, low_dim=low_dim, actions=action_array,
            success=success_array, collision=collision_array,
            front_camera_intrinsics=intrinsics, front_camera_extrinsics=extrinsics,
            camera_intrinsics=intrinsics, camera_extrinsics=extrinsics,
            coordinate_frame=np.asarray("world"), **metadata,
        )
    np.savez_compressed(root / "metadata.npz", low_dim=low_dim, actions=action_array,
                        success=success_array, collision=collision_array, **metadata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="open_drawer")
    ap.add_argument("--output", default="data/raw/rlbench_multiview_perturbed")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--noise-std", type=float, nargs="+", default=(0.03,))
    ap.add_argument("--gripper-delay", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    ap.add_argument("--allow-failed-nominal", action="store_true",
                    help="continue collecting variants when expert replay fails")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench import tasks as task_module

    mode = MoveArmThenGripper(JointPosition(absolute_mode=True), Discrete())
    env = Environment(mode, obs_config=_camera_config(args.image_size, args.cameras), headless=True)
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
                replay_succeeded = False
                for step, base in enumerate(expert):
                    action = base.copy()
                    if noise_std:
                        action[:7] += rng.normal(0.0, noise_std, 7).astype(np.float32)
                    if delay and step > 0:
                        action[7] = expert[max(0, step - delay)][7]
                    action[7] = float(action[7] >= 0.5)
                    try:
                        obs, reward, terminate = task.step(action)
                    except Exception:
                        break
                    executed.append(action)
                    observations.append(obs)
                    collisions.append(_environment_collision(task, args.task))
                    replay_succeeded = replay_succeeded or reward > 0.5
                    if terminate:
                        break
                succeeded = replay_succeeded
                destination = task_root / f"episode_{start_index + episode:04d}" / name
                _save_rollout(observations, executed, collisions, succeeded, destination,
                              {"task": np.asarray(args.task), "variant": np.asarray(name),
                               "noise_std": np.asarray(noise_std, dtype=np.float32),
                               "gripper_delay": np.asarray(delay, dtype=np.int32),
                               "planner_demo_success": np.asarray(1.0, dtype=np.float32),
                               "replay_success": np.asarray(float(replay_succeeded), dtype=np.float32)},
                              cameras=args.cameras)
                print(f"saved {destination} frames={len(observations)} success={int(succeeded)} "
                      f"collisions={int(sum(collisions))}", flush=True)
                if name == "expert" and not replay_succeeded and not args.allow_failed_nominal:
                    print("skip perturbations: nominal replay failed", flush=True)
                    break
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
