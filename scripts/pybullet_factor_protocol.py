"""Same action-geometry protocol as mujoco_strong_multitask, in PyBullet.

This is the second-simulator check: identical factorial layouts, labels, and
closed-loop selection, different physics engine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.mujoco_strong_multitask import (
    DIRS,
    classify,
    closed_loop,
    evaluate,
    feat,
)

TASKS = ("reach", "push", "pick_place", "insert")


def make_world():
    import pybullet as pb
    cid = pb.connect(pb.DIRECT)
    pb.setGravity(0, 0, -9.81)
    pb.setTimeStep(0.01)
    plane = pb.createCollisionShape(pb.GEOM_PLANE)
    pb.createMultiBody(0, plane)
    ee_col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.02)
    ee = pb.createMultiBody(1.0, ee_col, basePosition=[0, 0, 0.12])
    pay_col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.018)
    pay = pb.createMultiBody(0.03, pay_col, basePosition=[0, 0, 0.07])
    pb.changeDynamics(ee, -1, lateralFriction=1.0, spinningFriction=0.1)
    pb.changeDynamics(pay, -1, lateralFriction=1.0, spinningFriction=0.1)
    return pb, cid, ee, pay


def rollout(pb, ee, pay, task, direction, amplitude, payload_xy):
    start = np.array([payload_xy[0], payload_xy[1], 0.12], np.float64)
    if task == "push":
        start[:2] = np.asarray(payload_xy, np.float64) - DIRS[direction] * 0.05
        start[2] = 0.07
    quat = [0, 0, 0, 1]
    pb.resetBasePositionAndOrientation(ee, start.tolist(), quat)
    pb.resetBasePositionAndOrientation(pay, [payload_xy[0], payload_xy[1], 0.07], quat)
    pb.resetBaseVelocity(pay, [0, 0, 0], [0, 0, 0])
    end = start.copy()
    end[:2] += DIRS[direction] * amplitude
    for t in range(30):
        a = min(1.0, (t + 1) / 26.0)
        pos = start * (1 - a) + end * a
        if task == "push":
            cur, _ = pb.getBasePositionAndOrientation(ee)
            vel = (pos - np.asarray(cur, np.float64)) / 0.01
            pb.resetBaseVelocity(ee, vel.tolist(), [0, 0, 0])
        else:
            pb.resetBasePositionAndOrientation(ee, pos.tolist(), quat)
            if task in ("pick_place", "insert") and t < 24:
                carry = pos + np.array([0, 0, -0.05])
                pb.resetBasePositionAndOrientation(pay, carry.tolist(), quat)
                pb.resetBaseVelocity(pay, [0, 0, 0], [0, 0, 0])
        pb.stepSimulation()
    if task == "insert":
        pos, _ = pb.getBasePositionAndOrientation(ee)
        pos = np.array(pos, np.float64)
        for _ in range(12):
            pos = pos + np.array([0, 0, -0.006])
            pb.resetBasePositionAndOrientation(ee, pos.tolist(), quat)
            pb.resetBasePositionAndOrientation(pay, (pos + np.array([0, 0, -0.05])).tolist(), quat)
            pb.resetBaseVelocity(pay, [0, 0, 0], [0, 0, 0])
            pb.stepSimulation()
    if task in ("pick_place", "insert", "push"):
        p, _ = pb.getBasePositionAndOrientation(pay)
        return np.array(p, np.float64)
    p, _ = pb.getBasePositionAndOrientation(ee)
    return np.array(p, np.float64)


def paired_rows(n, seed=2027):
    rng = np.random.default_rng(seed)
    pb, cid, ee, pay = make_world()
    rows = []
    groups = max(1, n // (len(TASKS) * 3))
    try:
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
                    p = rollout(pb, ee, pay, task, d, 0.16, payload)
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
    finally:
        pb.disconnect(cid)
    return rows[:n]


def confounded_rows(n, seed=11):
    rng = np.random.default_rng(seed)
    pb, cid, ee, pay = make_world()
    rows = []
    target = np.array([0.16, 0.0])
    try:
        for i in range(n):
            amp = 0.08 if i % 2 else 0.16
            p = rollout(pb, ee, pay, "reach", 0, amp, np.zeros(2))
            label, yi = classify(p[:2], target, np.array([[0, 0.16], [0, -0.16]]))
            rows.append({
                "task": "reach", "direction": 0, "amplitude": amp,
                "label": label, "y": yi,
                "target_xy": target.tolist(), "wrong_xy": [0, 0.16, 0, -0.16],
                "payload_xy": [0, 0], "endpoint_xy": p[:2].tolist(),
            })
    finally:
        pb.disconnect(cid)
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
    tr_idx = [i for i, g in enumerate(groups) if g in tr_groups]
    te_idx = [i for i, g in enumerate(groups) if g not in tr_groups]
    train, test = [rows[i] for i in tr_idx], [rows[i] for i in te_idx]
    paired = evaluate(train, test, "paired")
    perturb = evaluate(conf, test, "perturbation")
    Xa = np.stack([feat(r, action=True)[:4] for r in train])
    Xat = np.stack([feat(r, action=True)[:4] for r in test])
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def gfeat(r):
        return np.r_[r["target_xy"], r["payload_xy"]]
    Xg = np.stack([gfeat(r) for r in train])
    Xgt = np.stack([gfeat(r) for r in test])
    y = np.array([r["y"] for r in train])
    yt = np.array([r["y"] for r in test])

    def au(X, Xt):
        c = LogisticRegression(max_iter=2000).fit(X, y)
        p = c.predict_proba(Xt)
        return float(roc_auc_score(yt, p, multi_class="ovr", average="macro"))

    out = {
        "simulator": "pybullet",
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
