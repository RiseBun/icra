"""Same action-geometry protocol in SAPIEN/PhysX (ManiSkill stack).

Isaac Sim does not fit on this machine (~15GB free). SAPIEN is the physics
backend behind ManiSkill/RoboTwin and is the realistic second engine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sapien
from sapien.physx import PhysxRigidDynamicComponent

from scripts.mujoco_strong_multitask import DIRS, classify, closed_loop, evaluate, feat


def _body(actor):
    return actor.find_component_by_type(PhysxRigidDynamicComponent)

TASKS = ("reach", "push", "pick_place", "insert")


def make_world():
    scene = sapien.Scene()
    scene.set_timestep(0.01)
    scene.add_ground(altitude=0.0)
    ee_b = scene.create_actor_builder()
    ee_b.add_sphere_collision(radius=0.012)
    ee_b.add_sphere_visual(radius=0.012)
    ee_b.set_initial_pose(sapien.Pose([0, 0, 0.12]))
    ee = ee_b.build_kinematic(name="ee")
    pay_b = scene.create_actor_builder()
    pay_b.add_sphere_collision(radius=0.018)
    pay_b.add_sphere_visual(radius=0.018)
    pay_b.set_initial_pose(sapien.Pose([0, 0, 0.07]))
    pay = pay_b.build(name="payload")
    return scene, ee, pay


def rollout(scene, ee, pay, task, direction, amplitude, payload_xy):
    start = np.array([payload_xy[0], payload_xy[1], 0.12], np.float64)
    ee.set_pose(sapien.Pose(start.tolist()))
    pay.set_pose(sapien.Pose([payload_xy[0], payload_xy[1], 0.07]))
    _body(pay).set_linear_velocity([0, 0, 0])
    _body(pay).set_angular_velocity([0, 0, 0])
    end = start.copy()
    end[:2] += DIRS[direction] * amplitude
    for t in range(30):
        a = min(1.0, (t + 1) / 26.0)
        pos = start * (1 - a) + end * a
        ee.set_pose(sapien.Pose(pos.tolist()))
        if task in ("pick_place", "insert") and t < 24:
            carry = pos + np.array([0.0, 0.0, -0.05])
            pay.set_pose(sapien.Pose(carry.tolist()))
            _body(pay).set_linear_velocity([0, 0, 0])
            _body(pay).set_angular_velocity([0, 0, 0])
        scene.step()
    if task == "insert":
        pos = np.array(ee.pose.p, np.float64)
        for _ in range(12):
            pos = pos + np.array([0.0, 0.0, -0.006])
            ee.set_pose(sapien.Pose(pos.tolist()))
            pay.set_pose(sapien.Pose((pos + np.array([0.0, 0.0, -0.05])).tolist()))
            _body(pay).set_linear_velocity([0, 0, 0])
            scene.step()
    if task in ("pick_place", "insert"):
        return np.array(pay.pose.p, np.float64)
    return np.array(ee.pose.p, np.float64)


def paired_rows(n, seed=2027):
    rng = np.random.default_rng(seed)
    scene, ee, pay = make_world()
    rows = []
    groups = max(1, n // (len(TASKS) * 3))
    for task in TASKS:
        for g in range(groups):
            payload = rng.uniform(-0.02, 0.02, 2)
            goal = int(rng.integers(3))
            wrong_dir = (goal + int(rng.integers(1, 3))) % 3
            drop_dir = 3 - goal - wrong_dir
            target = payload + DIRS[goal] * 0.16 + rng.uniform(-0.012, 0.012, 2)
            wrong1 = payload + DIRS[wrong_dir] * 0.16 + rng.uniform(-0.012, 0.012, 2)
            wrong2 = payload + DIRS[drop_dir] * 0.16 + np.array([0.075, 0.075])
            wrongs = np.vstack([wrong1, wrong2])
            for d in range(3):
                p = rollout(scene, ee, pay, task, d, 0.16, payload)
                label, yi = classify(p[:2], target, wrongs, pz=p[2], insert=(task == "insert"))
                rows.append({
                    "task": task, "direction": d, "amplitude": 0.16,
                    "label": label, "y": yi,
                    "target_xy": target.tolist(),
                    "wrong_xy": wrongs.reshape(-1).tolist(),
                    "payload_xy": payload.tolist(),
                    "endpoint_xy": p[:2].tolist(),
                    "endpoint_z": float(p[2]),
                    "layout_id": int(g),
                })
    return rows[:n]


def confounded_rows(n, seed=11):
    rng = np.random.default_rng(seed)
    scene, ee, pay = make_world()
    rows = []
    target = np.array([0.16, 0.0])
    for i in range(n):
        amp = 0.08 if i % 2 else 0.16
        p = rollout(scene, ee, pay, "reach", 0, amp, np.zeros(2))
        label, yi = classify(p[:2], target, np.array([[0, 0.16], [0, -0.16]]))
        rows.append({
            "task": "reach", "direction": 0, "amplitude": amp,
            "label": label, "y": yi,
            "target_xy": target.tolist(), "wrong_xy": [0, 0.16, 0, -0.16],
            "payload_xy": [0, 0], "endpoint_xy": p[:2].tolist(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = paired_rows(args.n)
    conf = confounded_rows(args.n)
    rng = np.random.default_rng(2028)
    groups = np.array([f"{r['task']}_{r.get('layout_id', i)}" for i, r in enumerate(rows)])
    ug = rng.permutation(np.unique(groups))
    gcut = len(ug) // 2
    tr_groups = set(ug[:gcut])
    train = [rows[i] for i, g in enumerate(groups) if g in tr_groups]
    test = [rows[i] for i, g in enumerate(groups) if g not in tr_groups]
    paired = evaluate(train, test, "paired")
    perturb = evaluate(conf, test, "perturbation")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Xa = np.stack([feat(r)[:4] for r in train])
    Xat = np.stack([feat(r)[:4] for r in test])
    Xg = np.stack([np.r_[r["target_xy"], r["payload_xy"]] for r in train])
    Xgt = np.stack([np.r_[r["target_xy"], r["payload_xy"]] for r in test])
    y = np.array([r["y"] for r in train])
    yt = np.array([r["y"] for r in test])

    def au(X, Xt):
        c = LogisticRegression(max_iter=2000).fit(X, y)
        return float(roc_auc_score(yt, c.predict_proba(Xt), multi_class="ovr", average="macro"))

    out = {
        "simulator": "sapien_physx",
        "n": len(rows),
        "tasks": {t: sum(r["task"] == t for r in rows) for t in TASKS},
        "class_counts": {k: sum(r["label"] == k for r in rows) for k in ("success", "wrong_target", "drop")},
        "paired_test": {
            "action_only_macro_auroc": au(Xa, Xat),
            "geometry_only_macro_auroc": au(Xg, Xgt),
            "paired_predictor": paired,
            "perturbation_predictor": perturb,
            "paired_minus_perturbation_auroc": paired["binary_success_auroc"] - perturb["binary_success_auroc"],
        },
        "closed_loop": closed_loop(train, test, conf),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
