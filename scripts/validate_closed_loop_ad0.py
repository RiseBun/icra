"""Closed-loop target insertion gate for one RLBench peg episode.

The controller repeatedly recomputes the tip target from the measured
ring-tip offset.  It is intentionally a feasibility diagnostic, not a
learned policy: the goal is to test whether execution noise, rather than
task geometry, caused the open-loop candidate failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_peg_grasp_events import expert_actions, find_grasp_boundary, make_environment
from validate_static_ad0 import shape, replay_to_grasp, get_snapshot, execute_and_trace


def target_from_color(variation: int):
    from rlbench.const import colors
    expected = np.asarray(colors[int(variation)][1], dtype=np.float32)
    cols = np.stack([np.asarray(shape(f"pillar{i}").get_color(), dtype=np.float32)
                     for i in range(3)])
    target = int(np.argmin(np.linalg.norm(cols - expected[None], axis=1)))
    return target, expected.tolist(), cols.tolist()


def _path_points(arm, goal, euler, steps=5):
    path = arm.get_linear_path(goal.tolist(), euler=euler.tolist(), steps=steps,
                               ignore_collisions=True)
    return path._path_points.reshape(-1, arm.get_joint_count()).astype(np.float32)


def closed_loop_insert(task, target, correction_steps=12, segment_steps=5,
                       release_steps=8, seat_z=0.05):
    arm = task._robot.arm
    tip = arm.get_tip()
    inner = getattr(task, "_task", task)
    ring_obj = inner._square_ring
    pillar = shape(f"pillar{target}")
    trace = []

    # Replan from measurements after every short segment.  The ring-tip offset
    # changes as the grasp settles, so a one-shot IK target is insufficient.
    for _ in range(correction_steps):
        ring = np.asarray(ring_obj.get_position(), dtype=np.float64)
        tip_pos = np.asarray(tip.get_position(), dtype=np.float64)
        offset = ring - tip_pos
        desired_ring = np.asarray(pillar.get_position(), dtype=np.float64).copy()
        desired_ring[2] += float(seat_z)
        desired_tip = desired_ring - offset
        euler = np.asarray(tip.get_orientation(), dtype=np.float64)
        points = _path_points(arm, desired_tip, euler, steps=segment_steps)
        for joints in points[1:]:
            action = np.zeros(8, dtype=np.float32)
            action[:7] = joints
            action[7] = 0.0
            task.step(action)
            trace.append(get_snapshot(task, target))
        if trace[-1]["target_xy"] < 0.008:
            break

    # A short vertical lift/descend removes lateral force before release.
    tip_pos = np.asarray(tip.get_position(), dtype=np.float64)
    euler = np.asarray(tip.get_orientation(), dtype=np.float64)
    for z_delta in (0.06, 0.0):
        goal = tip_pos.copy(); goal[2] += z_delta
        points = _path_points(arm, goal, euler, steps=6)
        for joints in points[1:]:
            action = np.zeros(8, dtype=np.float32)
            action[:7] = joints; action[7] = 0.0
            task.step(action)
            trace.append(get_snapshot(task, target))

    # Release and allow the ring to settle on the seat.
    hold = np.zeros(8, dtype=np.float32)
    hold[:7] = np.asarray(arm.get_joint_positions(), dtype=np.float32)
    hold[7] = 1.0
    rewards = []
    for _ in range(release_steps + 8):
        _, reward, terminate = task.step(hold)
        rewards.append(float(reward))
        trace.append(get_snapshot(task, target))
        if terminate:
            break
    stable = sum(1 for s in trace[-8:] if s["target_xy"] < 0.035)
    return {
        "reward_success": int(any(r > 0.5 for r in rewards)),
        "stable_target_steps": int(stable),
        "final": trace[-1],
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variations", type=int, nargs="+", default=(0,))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--output", default="data/diagnostics/closed_loop_ad0")
    ap.add_argument("--image-size", type=int, default=96)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variation in args.variations:
        for rep in range(args.episodes):
            env, task = make_environment(args.image_size)
            try:
                task.set_variation(int(variation))
                demo = task.get_demos(1, live_demos=True, max_attempts=10)[0]
                # Capture the target before any reset_to_demo call.  RLBench's
                # init_episode samples the chosen pillar randomly.
                target, expected_color, colors = target_from_color(variation)
                expert = expert_actions(demo)
                boundary = find_grasp_boundary(task, demo, expert)
                if boundary is None:
                    rows.append({"variation": variation, "rep": rep,
                                 "error": "no_grasp_boundary"}); continue
                settle = boundary + 6
                replay_to_grasp(task, demo, expert, settle)
                before = get_snapshot(task, target)
                result = closed_loop_insert(task, target)
                row = {"variation": int(variation), "rep": int(rep),
                       "target": int(target), "expected_color": expected_color,
                       "pillar_colors": colors, "grasp": before,
                       "reward_success": result["reward_success"],
                       "stable_target_steps": result["stable_target_steps"],
                       "final": result["final"]}
                rows.append(row)
                print(json.dumps(row, separators=(",", ":")), flush=True)
            except Exception as exc:
                row = {"variation": int(variation), "rep": int(rep),
                       "error": f"{type(exc).__name__}: {exc}"}
                rows.append(row); print(json.dumps(row), flush=True)
            finally:
                env.shutdown()
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
