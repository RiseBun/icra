"""Crossed-factor counterfactual collector: geometry x direction x D.

For each episode we build, from a single grasped state, three Cartesian-
equidistant insertion actions d0/d1/d2 (one per pillar direction) at each D,
then replay every (geometry, direction) pair and record the outcome plus the
scalar profile. Output is one JSONL of structured samples for the later
Action-only / Geometry-only / Action+Geometry AUROC evaluation.

geometry A: default layout
geometry B: target <-> wrong1 swap
geometry C: target <-> wrong2 swap
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_peg_grasp_events import expert_actions, make_environment, find_grasp_boundary
from peg_geometry_signals import extract_peg_signals


def build_actions(arm, tip_pos, tip_euler, pillars, D, steps=30,
                  release_steps=12):
    actions = {}
    for k, pillar in enumerate(pillars):
        pxy = np.asarray(pillar.get_position())[:2]
        dxy = pxy - tip_pos[:2]
        n = np.linalg.norm(dxy)
        if n < 1e-4:
            continue
        dir_k = dxy / n
        target = tip_pos.copy()
        target[:2] = tip_pos[:2] + D * dir_k
        # success requires the ring ON TOP of the pillar, which is ABOVE the
        # grasped height, so set the target z to the pillar top (not a descent).
        target[2] = float(pillar.get_position()[2]) + 0.05
        path = arm.get_linear_path(target.tolist(), euler=tip_euler.tolist(),
                                   steps=steps, ignore_collisions=True)
        pts = path._path_points.reshape(-1, arm.get_joint_count()).astype(np.float32)
        # Keep the approach scalar-matched, then add the same lift/descend/
        # release pattern for every direction.  Success in RLBench requires
        # the ring to be released and to settle on the proximity sensor.
        initial_joints = np.asarray(arm.get_joint_positions(), dtype=np.float32)
        arm.set_joint_positions(pts[-1], disable_dynamics=True)
        lift_target = target.copy(); lift_target[2] += 0.08
        lift_path = arm.get_linear_path(lift_target.tolist(), euler=tip_euler.tolist(),
                                        steps=max(3, release_steps // 3),
                                        ignore_collisions=True)
        lift_pts = lift_path._path_points.reshape(-1, arm.get_joint_count()).astype(np.float32)
        arm.set_joint_positions(lift_pts[-1], disable_dynamics=True)
        descend_path = arm.get_linear_path(target.tolist(), euler=tip_euler.tolist(),
                                           steps=max(3, release_steps // 3),
                                           ignore_collisions=True)
        descend_pts = descend_path._path_points.reshape(-1, arm.get_joint_count()).astype(np.float32)
        arm.set_joint_positions(initial_joints, disable_dynamics=True)
        suffix = np.concatenate((lift_pts[1:], descend_pts[1:]), axis=0)
        suffix = suffix[:max(1, release_steps)]
        if len(suffix) < release_steps:
            suffix = np.concatenate((suffix,
                                     np.repeat(suffix[-1:], release_steps - len(suffix), axis=0)), axis=0)
        chunk = np.zeros((len(pts) + release_steps, 8), dtype=np.float32)
        chunk[:len(pts), :7] = pts
        chunk[len(pts):, :7] = suffix
        chunk[:, 7] = 0.0  # keep gripper CLOSED to hold the grasped ring
        chunk[-4:, 7] = 1.0  # open and allow the ring to drop/settle
        actions[k] = chunk
    return actions


def replay_outcome(task, chunk):
    success = False
    end_wrong = end_target = None
    ever_grasped = bool(task._robot.gripper.get_grasped_objects())
    dropped = False
    min_wrong_over_target = float("inf")
    for action in chunk:
        _, reward, terminate = task.step(action)
        success = success or float(reward) > 0.5
        sig = extract_peg_signals(task)
        end_wrong = sig["wrong_target_distance"]
        end_target = sig["target_distance"]
        min_wrong_over_target = min(
            min_wrong_over_target,
            float(sig["wrong_target_distance"] - sig["target_distance"]),
        )
        grasped = bool(task._robot.gripper.get_grasped_objects())
        if ever_grasped and not grasped and not success:
            dropped = True
        ever_grasped = ever_grasped or grasped
        if terminate:
            break
    # Labels are mutually exclusive: successful release dominates any later
    # physical separation, then wrong-target dominates an early drop.
    wrong = bool((not success) and min_wrong_over_target < -0.02)
    dropped = bool((not success) and dropped and not wrong)
    return success, wrong, dropped


def ring_geometry_snapshot(task, target_pillar_index):
    """Return raw final ring/pillar geometry for threshold calibration.

    This deliberately does not use success_centre or success sensors.
    """
    from pyrep.objects.shape import Shape
    inner = getattr(task, "_task", task)
    ring = np.asarray(inner._square_ring.get_position(), dtype=np.float32)
    pillars = np.stack([
        np.asarray(Shape(f"pillar{i}").get_position(), dtype=np.float32)
        for i in range(3)
    ])
    distances = np.linalg.norm(pillars[:, :2] - ring[:2], axis=1)
    nearest = int(np.argmin(distances))
    return {
        "ring_x": float(ring[0]), "ring_y": float(ring[1]), "ring_z": float(ring[2]),
        "target_pillar_index": int(target_pillar_index),
        "target_xy_distance": float(distances[target_pillar_index]),
        "nearest_pillar_index": nearest,
        "nearest_xy_distance": float(distances[nearest]),
        "wrong_xy_distance": float(min(distances[i] for i in range(3)
                                        if i != target_pillar_index)),
        "pillar_z": float(pillars[target_pillar_index, 2]),
        "ring_grasped": bool(task._robot.gripper.get_grasped_objects()),
    }


def classify_ring_geometry(snapshot, *, xy_threshold=0.035):
    """Classify final ring state without RLBench reward/sensor semantics.

    A valid release can settle the ring at z=0.755--0.777, so height is not a
    reliable hard separator.  The primary label is the nearest actual pillar
    after release; z is retained in the raw snapshot for diagnostics.
    """
    if snapshot.get("ring_grasped", False):
        return "drop"
    if snapshot["nearest_xy_distance"] > xy_threshold:
        return "drop"
    if snapshot["nearest_pillar_index"] == snapshot["target_pillar_index"]:
        return "success"
    return "wrong_target"


def set_geometry(task, inner, geometry_id, tgt, w1, w2, circle_positions, succ_pos):
    succ = inner._success_centre
    tgt_p = task._robot  # placeholder, replaced below via Shape
    from pyrep.objects.shape import Shape
    tgt_p = Shape(f"pillar{tgt}")
    w1_p = Shape(f"pillar{w1}")
    w2_p = Shape(f"pillar{w2}")
    # Put all three physical pillars on the same-radius circle.  The geometry
    # factor is then only which pillar identity is assigned to each location.
    layouts = {
        "A": (tgt, w1, w2),
        "B": (w1, tgt, w2),
        "C": (w2, w1, tgt),
    }
    location_to_shape = layouts[geometry_id]
    shapes = {tgt: tgt_p, w1: w1_p, w2: w2_p}
    for location, shape_idx in enumerate(location_to_shape):
        shapes[shape_idx].set_position(circle_positions[location].tolist())
    target_location = location_to_shape.index(tgt)
    s = succ_pos.copy()
    s[:2] = circle_positions[target_location][:2]
    succ.set_position(s.tolist())


def geometry_features(task, tgt):
    from pyrep.objects.shape import Shape
    inner = getattr(task, "_task", task)
    tip = task._robot.arm.get_tip()
    tip_pos = np.asarray(tip.get_position())
    feats = {
        "tip_x": float(tip_pos[0]), "tip_y": float(tip_pos[1]),
        "ring_x": float(inner._square_ring.get_position()[0]),
        "ring_y": float(inner._square_ring.get_position()[1]),
        "target_id": int(tgt),
    }
    for k in range(3):
        p = np.asarray(Shape(f"pillar{k}").get_position())
        feats[f"pillar{k}_x"] = float(p[0])
        feats[f"pillar{k}_y"] = float(p[1])
    return feats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--output", default="data/features/cross_factor")
    ap.add_argument("--D", type=float, nargs="+", default=(0.05, 0.08, 0.10, 0.12))
    ap.add_argument("--image-size", type=int, default=96)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "samples.jsonl"

    env, task = make_environment(args.image_size)
    try:
        with jsonl.open("w", encoding="utf-8") as fh:
            for episode in range(args.episodes):
                task.sample_variation()
                demo = task.get_demos(1, live_demos=True, max_attempts=5)[0]
                expert = expert_actions(demo)
                boundary = find_grasp_boundary(task, demo, expert)
                if boundary is None:
                    continue
                settle = boundary + 6

                # baseline geometry A success gate
                def replay_to_grasped():
                    _, obs = task.reset_to_demo(demo)
                    for a in expert[:settle]:
                        obs, _, term = task.step(a)
                        if term:
                            return False
                    return True

                if not replay_to_grasped():
                    continue
                arm = task._robot.arm
                tip = arm.get_tip()
                tip_pos = np.asarray(tip.get_position())
                tip_euler = np.asarray(tip.get_orientation())
                sig = extract_peg_signals(task)
                tgt = sig["target_pillar_index"]
                w1 = (tgt + 1) % 3
                w2 = (tgt + 2) % 3
                inner = getattr(task, "_task", task)
                from pyrep.objects.shape import Shape
                succ_pos = np.asarray(inner._success_centre.get_position())
                pillars = [Shape(f"pillar{i}") for i in range(3)]

                # baseline success gate: expert chunk under geometry A must succeed
                expert_chunk = np.zeros((len(expert[settle:]), 8), dtype=np.float32)
                expert_chunk[:, :7] = expert[settle:][:, :7]
                expert_chunk[:, 7] = expert[settle:][:, 7]
                a_ok, _, _ = replay_outcome(task, expert_chunk)
                if not a_ok:
                    print(f"episode={episode}: baseline A failed -> gated", flush=True)
                    continue

                for D in args.D:
                    replay_to_grasped()
                    actions = build_actions(arm, tip_pos, tip_euler, pillars, D)
                    if len(actions) != 3:
                        continue
                    # Fixed-radius locations make both the Cartesian endpoints
                    # and the physical target geometry scalar-matched.
                    circle_positions = []
                    for pillar in pillars:
                        direction = np.asarray(pillar.get_position())[:2] - tip_pos[:2]
                        direction /= max(float(np.linalg.norm(direction)), 1e-8)
                        pos = tip_pos.copy()
                        pos[:2] = tip_pos[:2] + float(D) * direction
                        pos[2] = float(pillar.get_position()[2])
                        circle_positions.append(pos)
                    circle_positions = np.asarray(circle_positions, dtype=np.float32)
                    for geometry_id in ("A", "B", "C"):
                        for direction_id in sorted(actions):
                            try:
                                if not replay_to_grasped():
                                    continue
                                set_geometry(task, inner, geometry_id, tgt, w1, w2,
                                             circle_positions, succ_pos)
                                chunk = actions[direction_id]
                                # scalar profile of this action
                                jl2 = float(np.linalg.norm(np.diff(chunk[:, :7], axis=0), axis=-1).sum())
                                pv = float(np.linalg.norm(np.diff(chunk[:, :7], axis=0), axis=-1).max() / 0.05)
                                cart = D
                                reward_success, _, _ = replay_outcome(task, chunk)
                                snapshot = ring_geometry_snapshot(task, tgt)
                                geom_label = classify_ring_geometry(snapshot)
                            except Exception as exc:
                                print(f"episode={episode} D={D} geo={geometry_id} "
                                      f"dir={direction_id}: replay error {type(exc).__name__}", flush=True)
                                continue
                            sample = {
                                "episode_id": episode,
                                "D": D,
                                "geometry_id": geometry_id,
                                "direction_id": direction_id,
                                "target_pillar": tgt,
                                "success": int(geom_label == "success"),
                                "wrong_target": int(geom_label == "wrong_target"),
                                "drop": int(geom_label == "drop"),
                                "reward_success": int(reward_success),
                                "ring_snapshot": snapshot,
                                "cart_disp": cart,
                                "joint_l2": jl2,
                                "peak_vel": pv,
                                "geometry_features": geometry_features(task, tgt),
                            }
                            fh.write(json.dumps(sample) + "\n")
                print(f"episode={episode} done (D={list(args.D)})", flush=True)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
