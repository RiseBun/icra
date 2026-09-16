"""External-validity stress test across independent physics engines.

The primary AGCD result uses a clean factorial protocol.  This companion
experiment keeps the protocol but adds continuous target jitter and endpoint
execution noise.  It reports whether geometry still adds information beyond
the action direction and amplitude in MuJoCo, PyBullet, and SAPIEN/PhysX.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIRS = np.array([[1.0, 0.0], [-0.5, 0.8660254], [-0.5, -0.8660254]], np.float64)
TASKS = ("reach", "push", "pick_place", "insert")


def engine_api(name):
    if name == "mujoco":
        mod = importlib.import_module("mujoco_strong_multitask")
        mujoco, model, data = mod.make_world()
        return mod, (mujoco, model, data)
    if name == "pybullet":
        mod = importlib.import_module("pybullet_factor_protocol")
        pb, cid, ee, pay = mod.make_world()
        return mod, (pb, cid, ee, pay)
    if name == "sapien":
        mod = importlib.import_module("sapien_factor_protocol")
        return mod, mod.make_world()
    raise ValueError(name)


def close_engine(name, state):
    if name == "pybullet":
        state[0].disconnect(state[1])


def set_mass_scale(name, mod, state, scale):
    """Perturb payload mass using each engine's native API."""
    if name == "mujoco":
        # payload is the first dynamic body after world (body id 1).
        state[1].body_mass[1] = 0.03 * float(scale)
    elif name == "pybullet":
        state[0].changeDynamics(state[3], -1, mass=0.03 * float(scale))
    else:
        mod._body(state[2]).set_mass(0.03 * float(scale))


def rollout(name, mod, state, task, direction, amplitude, payload):
    if name == "mujoco":
        return mod.rollout(state[0], state[1], state[2], task, direction, amplitude, payload)
    if name == "pybullet":
        return mod.rollout(state[0], state[2], state[3], task, direction, amplitude, payload)
    return mod.rollout(state[0], state[1], state[2], task, direction, amplitude, payload)


def classify(mod, endpoint, target, wrongs, task):
    return mod.classify(
        endpoint[:2], target, wrongs, pz=float(endpoint[2]), insert=(task == "insert")
    )


def features(row, mode):
    action = np.eye(3, dtype=np.float64)[row["direction"]]
    if mode == "action":
        return np.r_[action, row["amplitude"]]
    geom = np.r_[row["target"], row["payload"]]
    if mode == "geometry":
        return geom
    return np.r_[action, row["amplitude"], geom]


def fit_eval(train, test, mode, bootstrap=500):
    x = np.stack([features(r, mode) for r in train])
    xt = np.stack([features(r, mode) for r in test])
    y = np.asarray([r["class"] for r in train], np.int64)
    yt = np.asarray([r["class"] for r in test], np.int64)
    clf = MLPClassifier(hidden_layer_sizes=(48, 32), max_iter=1000,
                        random_state=2027).fit(x, y)
    prob = clf.predict_proba(xt)
    pred = clf.classes_[np.argmax(prob, axis=1)]
    # At low endpoint noise the drop class can legitimately be absent from a
    # held-out split.  Compute macro OVR over classes that are present in both
    # the classifier and test set, and report the class support explicitly.
    result = {"macro_auroc": None, "accuracy": float(accuracy_score(yt, pred))}
    common = [int(c) for c in clf.classes_ if c in set(np.unique(yt))]
    result["auroc_classes"] = common
    if len(common) >= 2:
        aucs = []
        for c in common:
            col = int(np.flatnonzero(clf.classes_ == c)[0])
            y_bin = (yt == c).astype(int)
            if len(np.unique(y_bin)) == 2:
                aucs.append(roc_auc_score(y_bin, prob[:, col]))
        if aucs:
            result["macro_auroc"] = float(np.mean(aucs))
            rng = np.random.default_rng(7027)
            boots = []
            for _ in range(bootstrap):
                idx = rng.integers(0, len(yt), len(yt))
                yb, pb = yt[idx], prob[idx]
                vals = []
                for c in common:
                    col = int(np.flatnonzero(clf.classes_ == c)[0])
                    yc = (yb == c).astype(int)
                    if len(np.unique(yc)) == 2:
                        vals.append(roc_auc_score(yc, pb[:, col]))
                if vals:
                    boots.append(float(np.mean(vals)))
            if boots:
                result["macro_auroc_ci95"] = [
                    float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5)),
                ]
    success_col = int(np.flatnonzero(clf.classes_ == 2)[0]) if 2 in clf.classes_ else None
    if success_col is not None and len(np.unique(yt)) > 1:
        result["binary_success_auroc"] = float(
            roc_auc_score((yt == 2).astype(int), prob[:, success_col])
        )
    return result


def confounded_amp(name, mod, state, n, rng, endpoint_noise):
    rows = []
    target = np.array([0.16, 0.0], np.float64)
    wrongs = np.array([[0.0, 0.16], [0.0, -0.16]], np.float64)
    for i in range(n):
        amplitude = 0.08 if i % 2 else 0.16
        p = np.asarray(rollout(name, mod, state, "reach", 0, amplitude,
                               np.zeros(2)), np.float64)
        p[:2] += rng.normal(0.0, endpoint_noise, 2)
        label, _ = classify(mod, p, target, wrongs, "reach")
        rows.append({"amplitude": amplitude, "success": int(label == "success")})
    # Larger amplitude is the shortcut in this constructed rollout set:
    # it reaches the target while the smaller amplitude misses it.
    score = np.asarray([r["amplitude"] for r in rows])
    y = np.asarray([r["success"] for r in rows])
    return float(roc_auc_score(y, score))


def paired_data(name, mod, state, n_layouts, rng, endpoint_noise, mass_scale=1.0):
    rows = []
    set_mass_scale(name, mod, state, mass_scale)
    for layout in range(n_layouts):
        task = TASKS[layout % len(TASKS)]
        payload = rng.uniform(-0.025, 0.025, 2)
        goal = int(rng.integers(3))
        target = payload + DIRS[goal] * 0.16 + rng.uniform(-0.022, 0.022, 2)
        wrong_dirs = [(goal + 1) % 3, (goal + 2) % 3]
        wrongs = np.stack([
            payload + DIRS[d] * 0.16 + rng.uniform(-0.022, 0.022, 2)
            for d in wrong_dirs
        ])
        for direction in range(3):
            p = np.asarray(rollout(name, mod, state, task, direction, 0.16, payload), np.float64)
            p[:2] += rng.normal(0.0, endpoint_noise, 2)
            label, cls = classify(mod, p, target, wrongs, task)
            rows.append({
                "layout": layout,
                "task": task,
                "direction": direction,
                "amplitude": 0.16,
                "target": target.tolist(),
                "payload": payload.tolist(),
                "label": label,
                "class": int(cls),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layouts", type=int, default=200)
    ap.add_argument("--endpoint-noise", type=float, default=0.006)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--mass-scale", type=float, default=1.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    reports = {}
    engine_rows = {}
    engine_splits = {}
    for name in ("mujoco", "pybullet", "sapien"):
        rng = np.random.default_rng(args.seed)
        mod, state = engine_api(name)
        try:
            conf_auc = confounded_amp(name, mod, state, args.n_layouts * 2, rng, args.endpoint_noise)
            rows = paired_data(name, mod, state, args.n_layouts, rng, args.endpoint_noise, args.mass_scale)
            layouts = np.array([r["layout"] for r in rows])
            split = set(np.unique(layouts)[::2])
            train = [r for r in rows if r["layout"] not in split]
            test = [r for r in rows if r["layout"] in split]
            engine_rows[name] = rows
            engine_splits[name] = (train, test)
            result = {
                "n_layouts": args.n_layouts,
                "n_rows": len(rows),
                "class_counts": {k: int(sum(r["label"] == k for r in rows))
                                 for k in ("success", "wrong_target", "drop")},
                "confounded_amp_auroc": conf_auc,
                "action_only": fit_eval(train, test, "action"),
                "geometry_only": fit_eval(train, test, "geometry"),
                "action_geometry": fit_eval(train, test, "joint"),
                "endpoint_noise_m": args.endpoint_noise,
                "mass_scale": args.mass_scale,
            }
            if (result["action_geometry"]["macro_auroc"] is not None
                    and result["action_only"]["macro_auroc"] is not None):
                result["joint_minus_action"] = (
                    result["action_geometry"]["macro_auroc"]
                    - result["action_only"]["macro_auroc"]
                )
            else:
                result["joint_minus_action"] = None
            reports[name] = result
        finally:
            close_engine(name, state)
    # Cross-engine transfer: fit on odd layouts from one engine and evaluate
    # on even layouts from another.  This is the stronger external-validity
    # check; no labels or features are shared at evaluation time.
    transfer = {}
    for source in engine_splits:
        transfer[source] = {}
        train_source = engine_splits[source][0]
        for target in engine_splits:
            if source == target:
                continue
            test_target = engine_splits[target][1]
            transfer[source][target] = fit_eval(train_source, test_target, "joint")
    reports["cross_engine_transfer"] = transfer
    engines = [e for e in reports if e in engine_splits]
    reports["summary"] = {
        "mean_amp_auroc": float(np.mean([reports[e]["confounded_amp_auroc"] for e in engines])),
        "mean_joint_minus_action": float(np.mean([reports[e]["joint_minus_action"] for e in engines if reports[e]["joint_minus_action"] is not None])),
        "engines": engines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
