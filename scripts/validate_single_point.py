"""Validate the minimum success cell: geometry A, direction d0.

The generated path is anchored on the grasped ring pose rather than the arm
tip pose.  RLBench evaluates insertion after release, so the ring-tip offset
must be preserved when constructing the tip endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_peg_grasp_events import expert_actions, make_environment, find_grasp_boundary
from peg_geometry_signals import extract_peg_signals


def _shape(name):
    from pyrep.objects.shape import Shape
    return Shape(name)


def replay_to_grasp(task, demo, expert, settle):
    _, obs = task.reset_to_demo(demo)
    for action in expert[:settle]:
        obs, _, terminate = task.step(action)
        if terminate:
            raise RuntimeError("demo terminated before grasp boundary")


def snapshot(task, target):
    inner = getattr(task, "_task", task)
    ring = np.asarray(inner._square_ring.get_position(), dtype=np.float64)
    tip = np.asarray(task._robot.arm.get_tip().get_position(), dtype=np.float64)
    pillars = np.stack([np.asarray(_shape(f"pillar{i}").get_position(), dtype=np.float64)
                        for i in range(3)])
    dist = np.linalg.norm(pillars[:, :2] - ring[:2], axis=1)
    return {
        "ring": ring.tolist(),
        "tip": tip.tolist(),
        "ring_tip_offset": (ring - tip).tolist(),
        "pillars": pillars.tolist(),
        "target": int(target),
        "target_xy": float(dist[target]),
        "nearest": int(np.argmin(dist)),
        "nearest_xy": float(dist.min()),
        "grasped": bool(task._robot.gripper.get_grasped_objects()),
    }


def set_and_assert_geometry(task, target, requested):
    """Apply geometry A and fail fast if simulator poses do not match."""
    for i, pos in enumerate(requested):
        _shape(f"pillar{i}").set_position(np.asarray(pos, dtype=np.float64).tolist())
    actual = np.stack([np.asarray(_shape(f"pillar{i}").get_position(), dtype=np.float64)
                       for i in range(3)])
    err = np.linalg.norm(actual - requested, axis=1)
    if float(err.max()) > 2e-3:
        raise RuntimeError(f"geometry assertion failed: max position error={err.max():.6f}")
    return actual


def build_ring_anchored_action(task, target_pos, *, steps=30, release_steps=12):
    arm = task._robot.arm
    tip = arm.get_tip()
    tip_pos = np.asarray(tip.get_position(), dtype=np.float64)
    tip_euler = np.asarray(tip.get_orientation(), dtype=np.float64)
    inner = getattr(task, "_task", task)
    ring_pos = np.asarray(inner._square_ring.get_position(), dtype=np.float64)
    ring_tip_offset = ring_pos - tip_pos

    # Move the ring to the pillar center while preserving the grasp offset.
    desired_ring = np.asarray(target_pos, dtype=np.float64).copy()
    desired_ring[2] += 0.05
    desired_tip = desired_ring - ring_tip_offset
    path = arm.get_linear_path(desired_tip.tolist(), euler=tip_euler.tolist(),
                               steps=steps, ignore_collisions=True)
    pts = path._path_points.reshape(-1, arm.get_joint_count()).astype(np.float32)

    initial = np.asarray(arm.get_joint_positions(), dtype=np.float32)
    arm.set_joint_positions(pts[-1], disable_dynamics=True)
    lift = desired_tip.copy(); lift[2] += 0.08
    lp = arm.get_linear_path(lift.tolist(), euler=tip_euler.tolist(), steps=4,
                             ignore_collisions=True)
    lift_pts = lp._path_points.reshape(-1, arm.get_joint_count()).astype(np.float32)
    arm.set_joint_positions(lift_pts[-1], disable_dynamics=True)
    dp = arm.get_linear_path(desired_tip.tolist(), euler=tip_euler.tolist(), steps=4,
                             ignore_collisions=True)
    descend_pts = dp._path_points.reshape(-1, arm.get_joint_count()).astype(np.float32)
    arm.set_joint_positions(initial, disable_dynamics=True)

    suffix = np.concatenate((lift_pts[1:], descend_pts[1:]), axis=0)
    suffix = suffix[:release_steps]
    if len(suffix) < release_steps:
        suffix = np.concatenate((suffix, np.repeat(suffix[-1:], release_steps-len(suffix), axis=0)))
    chunk = np.zeros((len(pts) + release_steps, 8), dtype=np.float32)
    chunk[:len(pts), :7] = pts
    chunk[len(pts):, :7] = suffix
    chunk[:, 7] = 0.0
    chunk[-4:, 7] = 1.0
    return chunk, ring_tip_offset


def execute(task, chunk):
    reward_success = False
    for action in chunk:
        _, reward, terminate = task.step(action)
        reward_success = reward_success or float(reward) > 0.5
        if terminate:
            break
    return bool(reward_success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/diagnostics/single_point_calibration")
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env, task = make_environment(args.image_size)
    rows = []
    try:
        task.sample_variation()
        demo = task.get_demos(1, live_demos=True, max_attempts=5)[0]
        expert = expert_actions(demo)
        boundary = find_grasp_boundary(task, demo, expert)
        if boundary is None:
            raise RuntimeError("no grasp boundary")
        settle = boundary + 6
        replay_to_grasp(task, demo, expert, settle)
        target = int(extract_peg_signals(task)["target_pillar_index"])
        target_pos = np.asarray(_shape(f"pillar{target}").get_position(), dtype=np.float64)
        base = snapshot(task, target)

        # The target is already geometry A.  Assert its current coordinates,
        # then test the ring-anchored d0 action over release timing variants.
        requested = np.stack([np.asarray(_shape(f"pillar{i}").get_position(), dtype=np.float64)
                              for i in range(3)])
        set_and_assert_geometry(task, target, requested)
        for release_steps in (8, 12, 16):
            replay_to_grasp(task, demo, expert, settle)
            set_and_assert_geometry(task, target, requested)
            chunk, offset = build_ring_anchored_action(task, target_pos,
                                                       release_steps=release_steps)
            ok = execute(task, chunk)
            final = snapshot(task, target)
            rows.append({"release_steps": release_steps, "reward_success": int(ok),
                         "base": base, "ring_tip_offset": offset.tolist(),
                         "final": final})
            print(json.dumps(rows[-1], separators=(",", ":")), flush=True)
        (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
