"""Episode-level static-layout factor experiment.

Each geometry is configured before demonstration generation: the RLBench target
shape is assigned to one of three fixed, equal-radius seats P0/P1/P2. No shape
identity is changed after the demo starts. d0/d1/d2 are generated in the
resulting grasp state toward the same three seats and evaluated with a
geometry-stability label.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_peg_grasp_events import expert_actions, make_environment, find_grasp_boundary
from peg_geometry_signals import extract_peg_signals
from validate_static_ad0 import (shape, get_snapshot, replay_to_grasp,
                                  build_d0, execute_and_trace)


def configure_static_layout(task, target_shape, target_location, radius=0.17):
    """Place shape identities at equal-radius seats before get_demos()."""
    inner = getattr(task, "_task", task)
    tip = np.asarray(task._robot.arm.get_tip().get_position(), dtype=np.float64)
    angles = np.deg2rad(np.asarray([0.0, 120.0, 240.0]))
    seats = np.zeros((3, 3), dtype=np.float64)
    seats[:, :2] = tip[:2] + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    seats[:, 2] = np.asarray(shape("pillar0").get_position())[2]
    wrong = [i for i in range(3) if i != target_shape]
    location_to_shape = list(wrong)
    location_to_shape.insert(int(target_location), int(target_shape))
    for location, shape_idx in enumerate(location_to_shape):
        shape(f"pillar{shape_idx}").set_position(seats[location].tolist())
    succ = np.asarray(inner._success_centre.get_position(), dtype=np.float64)
    succ[:2] = seats[target_location, :2]
    inner._success_centre.set_position(succ.tolist())
    actual = np.stack([np.asarray(shape(f"pillar{i}").get_position(), dtype=np.float64)
                       for i in range(3)])
    expected = np.zeros_like(actual)
    for location, shape_idx in enumerate(location_to_shape):
        expected[shape_idx] = seats[location]
    error = float(np.linalg.norm(actual - expected, axis=1).max())
    if error > 2e-3:
        raise RuntimeError(f"static layout assertion failed: {error:.6f} m")
    # reset_to_demo() restores the scene's initial task state before calling
    # init_episode().  Persist the configured geometry there so every replay
    # sees exactly the same pillar poses; runtime pose edits are invalid for
    # this factor experiment because they are lost during reset.
    scene = getattr(task, "_scene", None)
    if scene is not None and hasattr(scene, "get_state"):
        scene._initial_task_state = scene.get_state()
    return seats, location_to_shape, error


def apply_static_layout(task, target_shape, target_location, seats, mapping):
    """Restore the same static scene after reset_to_demo()."""
    inner = getattr(task, "_task", task)
    for location, shape_idx in enumerate(mapping):
        shape(f"pillar{shape_idx}").set_position(np.asarray(seats[location]).tolist())
    succ = np.asarray(inner._success_centre.get_position(), dtype=np.float64)
    succ[:2] = np.asarray(seats[target_location])[:2]
    inner._success_centre.set_position(succ.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-geometry", type=int, default=1)
    ap.add_argument("--geometries", nargs="+", default=("A", "B", "C"))
    ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--output", default="data/diagnostics/static_factor")
    ap.add_argument("--image-size", type=int, default=96)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env, task = make_environment(args.image_size)
    rows = []
    try:
        geometry_locations = {"A": 0, "B": 1, "C": 2}
        for geometry_id in args.geometries:
            target_location = geometry_locations[geometry_id]
            for rep in range(args.episodes_per_geometry):
                task.sample_variation(); task.reset()
                target_shape = int(extract_peg_signals(task)["target_pillar_index"])
                seats, mapping, layout_error = configure_static_layout(
                    task, target_shape, target_location, args.radius)
                demo = task.get_demos(1, live_demos=True, max_attempts=5)[0]
                expert = expert_actions(demo)
                boundary = find_grasp_boundary(task, demo, expert)
                if boundary is None:
                    rows.append({"geometry": geometry_id, "rep": rep,
                                 "error": "no_grasp_boundary"}); continue
                settle = boundary + 6
                apply_static_layout(task, target_shape, target_location, seats, mapping)
                replay_to_grasp(task, demo, expert, settle)
                # Build directions from the three fixed seats, then replay each
                # from the same grasp state. target_location is the expected
                # success direction for this geometry.
                pillars = [shape(f"pillar{i}") for i in range(3)]
                ordered = [pillars[idx] for idx in mapping]
                action_chunks = {}
                for d, pillar in enumerate(ordered):
                    # build_d0 is a single-direction helper; call it with a
                    # temporary target identity and preserve the current seat.
                    try:
                        action_chunks[d], _ = build_d0(task, mapping[d])
                    except Exception as exc:
                        action_chunks[d] = None
                        print(f"geometry={geometry_id} rep={rep} dir={d}: IK {type(exc).__name__}", flush=True)
                for direction_id in range(3):
                    if action_chunks[direction_id] is None:
                        rows.append({"geometry": geometry_id, "rep": rep,
                                     "direction_id": direction_id,
                                     "error": "ik_unreachable"})
                        continue
                    # Recreate grasp state before every replay.  Geometry was
                    # persisted in the scene initial state above, so no
                    # runtime shape mutation is performed here.
                    replay_to_grasp(task, demo, expert, settle)
                    result = execute_and_trace(task, action_chunks[direction_id], target_shape)
                    row = {
                        "geometry": geometry_id, "rep": rep,
                        "target_shape": target_shape,
                        "target_location": target_location,
                        "direction_id": direction_id,
                        "expected": int(direction_id == target_location),
                        "layout_error": layout_error,
                        "radius": args.radius,
                        "reward_success": result["reward_success"],
                        "stable_target_steps": result["stable_target_steps"],
                        "final": result["final"],
                    }
                    rows.append(row); print(json.dumps(row, separators=(",", ":")), flush=True)
        (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
