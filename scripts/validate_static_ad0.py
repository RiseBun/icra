"""Static-layout gate for the A/d0 insertion cell.

Pillar identities and poses are fixed before demo generation.  The test only
asks whether a target-directed, ring-anchored action can repeatedly release
the ring into the target seat; no runtime shape swapping is performed.
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


def shape(name):
    from pyrep.objects.shape import Shape
    return Shape(name)


def get_snapshot(task, target):
    inner = getattr(task, "_task", task)
    ring = np.asarray(inner._square_ring.get_position(), dtype=np.float64)
    tip = np.asarray(task._robot.arm.get_tip().get_position(), dtype=np.float64)
    pillars = np.stack([np.asarray(shape(f"pillar{i}").get_position(), dtype=np.float64)
                        for i in range(3)])
    d = np.linalg.norm(pillars[:, :2] - ring[:2], axis=1)
    return {
        "ring": ring.tolist(), "tip": tip.tolist(),
        "ring_tip_offset": (ring - tip).tolist(),
        "target": int(target), "target_xy": float(d[target]),
        "nearest": int(np.argmin(d)), "nearest_xy": float(d.min()),
        "ring_grasped": bool(task._robot.gripper.get_grasped_objects()),
    }


def set_static_layout(task, target):
    """Freeze current poses before demos are generated and assert no drift."""
    poses = np.stack([np.asarray(shape(f"pillar{i}").get_position(), dtype=np.float64)
                      for i in range(3)])
    for i, pos in enumerate(poses):
        shape(f"pillar{i}").set_position(pos.tolist())
    actual = np.stack([np.asarray(shape(f"pillar{i}").get_position(), dtype=np.float64)
                       for i in range(3)])
    err = np.linalg.norm(actual - poses, axis=1)
    if float(err.max()) > 2e-3:
        raise RuntimeError(f"static layout assertion failed: {err.max():.6f} m")
    return poses, float(err.max())


def replay_to_grasp(task, demo, expert, settle):
    _, obs = task.reset_to_demo(demo)
    for action in expert[:settle]:
        obs, _, terminate = task.step(action)
        if terminate:
            raise RuntimeError("demo terminated before grasp")


def build_d0(task, target, steps=30, release_steps=12):
    arm = task._robot.arm
    tip = arm.get_tip()
    tip_euler = np.asarray(tip.get_orientation(), dtype=np.float64)
    tip_pos = np.asarray(tip.get_position(), dtype=np.float64)
    inner = getattr(task, "_task", task)
    ring = np.asarray(inner._square_ring.get_position(), dtype=np.float64)
    ring_tip_offset = ring - tip_pos
    pillar = shape(f"pillar{target}")
    desired_ring = np.asarray(pillar.get_position(), dtype=np.float64).copy()
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
    suffix = np.concatenate((lift_pts[1:], descend_pts[1:]), axis=0)[:release_steps]
    if len(suffix) < release_steps:
        suffix = np.concatenate((suffix, np.repeat(suffix[-1:], release_steps-len(suffix), axis=0)))
    chunk = np.zeros((len(pts) + release_steps, 8), dtype=np.float32)
    chunk[:len(pts), :7] = pts
    chunk[len(pts):, :7] = suffix
    chunk[:, 7] = 0.0
    chunk[-4:, 7] = 1.0
    return chunk, ring_tip_offset


def execute_and_trace(task, chunk, target, settle_steps=8):
    rewards = []
    trace = []
    for action in chunk:
        _, reward, terminate = task.step(action)
        rewards.append(float(reward))
        trace.append(get_snapshot(task, target))
        if terminate:
            break
    # Continue stepping with an open gripper to measure whether the ring seat
    # remains stable after the release event.
    hold = chunk[-1].copy(); hold[7] = 1.0
    for _ in range(settle_steps):
        _, reward, terminate = task.step(hold)
        rewards.append(float(reward))
        trace.append(get_snapshot(task, target))
        if terminate:
            break
    final = trace[-1]
    stable = sum(1 for s in trace[-settle_steps:] if s["target_xy"] < 0.035)
    return {
        "reward_success": int(any(r > 0.5 for r in rewards)),
        "final": final, "stable_target_steps": int(stable),
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--output", default="data/diagnostics/static_ad0_gate")
    ap.add_argument("--image-size", type=int, default=96)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env, task = make_environment(args.image_size)
    rows = []
    try:
        for episode in range(args.episodes):
            task.sample_variation()
            # TaskEnvironment materializes the task scene on reset; doing this
            # before get_demos lets us freeze poses without any runtime shape
            # identity swaps.
            task.reset()
            target = int(extract_peg_signals(task)["target_pillar_index"])
            frozen_poses, layout_error = set_static_layout(task, target)
            demo = task.get_demos(1, live_demos=True, max_attempts=5)[0]
            expert = expert_actions(demo)
            boundary = find_grasp_boundary(task, demo, expert)
            if boundary is None:
                rows.append({"episode": episode, "error": "no_grasp_boundary"})
                continue
            settle = boundary + 6
            replay_to_grasp(task, demo, expert, settle)
            before = get_snapshot(task, target)
            chunk, offset = build_d0(task, target)
            result = execute_and_trace(task, chunk, target)
            row = {
                "episode": episode, "target": target,
                "layout_error": layout_error,
                "frozen_poses": frozen_poses.tolist(),
                "grasp_snapshot": before,
                "build_ring_tip_offset": offset.tolist(),
                "reward_success": result["reward_success"],
                "stable_target_steps": result["stable_target_steps"],
                "final": result["final"],
            }
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)
        (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
