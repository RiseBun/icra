"""Cartesian-equidistant multi-direction IK construction + distance scan.

For each D in the scan, build 3 fixed insertion actions that start from the
SAME tip pose and move the SAME Cartesian distance D along the 3 pillar
directions (xy), keeping z and gripper fixed. Then audit the scalar profile:
cartesian displacement, joint L2, peak joint velocity, action length, gripper
profile, and endpoint target/wrong distances. The goal is the SMALLEST D that
still produces mixed geometric outcomes, with joint-L2 variation < 5-10%.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_peg_grasp_events import expert_actions, make_environment, find_grasp_boundary
from peg_geometry_signals import extract_peg_signals


def build_and_measure(arm, tip_pos, tip_euler, pillars, D, steps, gripper_val):
    records = []
    for k, pillar in enumerate(pillars):
        pxy = np.asarray(pillar.get_position())[:2]
        dxy = pxy - tip_pos[:2]
        n = np.linalg.norm(dxy)
        if n < 1e-4:
            records.append({"pillar": k, "error": "tip_over_pillar"})
            continue
        dir_k = dxy / n
        target = tip_pos.copy()
        target[:2] = tip_pos[:2] + D * dir_k
        try:
            path = arm.get_linear_path(target.tolist(), euler=tip_euler.tolist(),
                                       steps=steps, ignore_collisions=True)
        except Exception as exc:
            records.append({"pillar": k, "error": f"{type(exc).__name__}: {exc}"})
            continue
        pts = path._path_points.reshape(-1, arm.get_joint_count())
        # measure
        arm.set_joint_positions(pts[-1], disable_dynamics=True)
        end_tip = np.asarray(arm.get_tip().get_position())
        cart_disp = float(np.linalg.norm(end_tip[:2] - tip_pos[:2]))
        joint_l2 = float(np.linalg.norm(np.diff(pts, axis=0), axis=-1).sum())
        # peak joint velocity, dt = 0.05 (CoppeliaSim default)
        peak_vel = float(np.linalg.norm(np.diff(pts, axis=0), axis=-1).max() / 0.05)
        arm.set_joint_positions(pts[0], disable_dynamics=True)
        records.append({
            "pillar": k, "cart_disp": cart_disp, "joint_l2": joint_l2,
            "peak_vel": peak_vel, "length": len(pts),
        })
    return records


def main():
    from pyrep.objects.shape import Shape

    env, task = make_environment(96)
    try:
        task.sample_variation()
        demo = task.get_demos(1, live_demos=True, max_attempts=5)[0]
        expert = expert_actions(demo)
        boundary = find_grasp_boundary(task, demo, expert)
        settle = boundary + 6
        _, obs = task.reset_to_demo(demo)
        for action in expert[:settle]:
            obs, _, terminate = task.step(action)
            if terminate:
                break

        arm = task._robot.arm
        tip = arm.get_tip()
        tip_pos = np.asarray(tip.get_position())
        tip_euler = np.asarray(tip.get_orientation())
        sig = extract_peg_signals(task)
        tgt = sig["target_pillar_index"]
        pillars = [Shape(f"pillar{i}") for i in range(3)]
        print(f"tip_pos={tip_pos.tolist()} target_pillar={tgt}")

        for D in (0.05, 0.08, 0.10, 0.12):
            records = build_and_measure(arm, tip_pos, tip_euler, pillars, D, steps=30, gripper_val=0.0)
            print(f"\n=== D={D} ===")
            cart = [r["cart_disp"] for r in records if "cart_disp" in r]
            jl2 = [r["joint_l2"] for r in records if "joint_l2" in r]
            pv = [r["peak_vel"] for r in records if "peak_vel" in r]
            if not cart:
                print("  all IK failed")
                continue
            jl2_mean = float(np.mean(jl2))
            jl2_var = float(np.std(jl2) / jl2_mean) if jl2_mean > 0 else float("nan")
            print(f"  cart_disp: mean={np.mean(cart):.4f} std={np.std(cart):.5f} (per={[round(c,3) for c in cart]})")
            print(f"  joint_l2 : mean={jl2_mean:.4f} variation={jl2_var*100:.1f}% (per={[round(j,3) for j in jl2]})")
            print(f"  peak_vel : mean={np.mean(pv):.3f} std={np.std(pv):.3f}")
            print(f"  length   : {set(r['length'] for r in records if 'length' in r)}")
            for r in records:
                if "error" in r:
                    print(f"  pillar{r['pillar']}: ERROR {r['error']}")
                else:
                    mark = "T" if r["pillar"] == tgt else " "
                    print(f"  pillar{r['pillar']}[{mark}]: cart={r['cart_disp']:.4f} l2={r['joint_l2']:.4f} "
                          f"pv={r['peak_vel']:.3f} len={r['length']}")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
