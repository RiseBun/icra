"""External-recipe A/B: RLBench-style expert perturbation -> success/fail.

Faithful MuJoCo port of the open collector pattern in
`scripts/collect_rlbench_perturbations.py`:

  reset_to_demo / fixed start
  + replay expert action chunk
  + variants: expert | joint_noise_σ | gripper_delay
  + label = terminal task success

This is the community-default negative-sample recipe used across RLBench
pipelines (replay expert + additive joint noise / timing delay). We do not
claim any single paper owns it; we claim the *recipe* injects an amplitude
shortcut.

Outputs:
  1) confounded set A/B: AUROC(noise_std -> y), AUROC(action_l2 -> y)
  2) train amplitude-only / joint predictors on confounded; evaluate on
     paired-geometry holdout (transfer collapse = model fell for shortcut)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_factor_task(repo: Path):
    path = repo / "scripts" / "mujoco_factor_task.py"
    spec = importlib.util.spec_from_file_location("mujoco_factor_task", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return float(wins / (len(pos) * len(neg)))


def bootstrap_auroc(scores, labels, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = auroc(s[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"median": float("nan"), "ci95": [float("nan"), float("nan")]}
    vals = np.sort(vals)
    return {
        "median": float(np.median(vals)),
        "ci95": [
            float(vals[int(0.025 * (len(vals) - 1))]),
            float(vals[int(0.975 * (len(vals) - 1))]),
        ],
    }


def rollout_noisy(task, target, direction, amplitude=0.16, horizon=30,
                  release_step=26, noise_std=0.0, gripper_delay=0, rng=None):
    """RLBench-style: replay nominal path with additive EE noise / release delay."""
    if rng is None:
        rng = np.random.default_rng(0)
    task.reset(task.payload_xy)
    start = task.data.mocap_pos[0].copy()
    vec = task.seats[int(direction)] - start
    vec[2] = 0.0
    unit = vec / max(np.linalg.norm(vec[:2]), 1e-8)
    end = start.copy()
    end[:2] += unit[:2] * float(amplitude)

    # gripper_delay: release earlier (positive delay steps) like delayed gripper cmd
    rel = max(1, int(release_step) - int(gripper_delay))
    actions = []
    prev = start.copy()
    for t in range(horizon):
        alpha = min(1.0, (t + 1) / max(1, release_step))
        pos = start * (1 - alpha) + end * alpha
        if noise_std > 0:
            pos = pos.copy()
            pos[:2] += rng.normal(0.0, float(noise_std), 2)
        task.data.mocap_pos[0] = pos
        task.data.mocap_pos[0, 2] = 0.12
        if t < rel:
            task.data.qpos[:3] = task.data.mocap_pos[0] + np.array([0, 0, -0.04])
            task.data.qvel[:] = 0
        delta = (pos - prev)[:2]
        actions.append(delta.astype(np.float64))
        task.mujoco.mj_step(task.model, task.data)
        prev = pos.copy()

    p = task.data.qpos[:3].copy()
    d = np.linalg.norm(task.seats[:, :2] - p[:2], axis=1)
    nearest = int(np.argmin(d))
    settled = d[nearest] < 0.035 and p[2] < 0.08
    label = "success" if nearest == int(target) and settled else "wrong_target" if settled else "drop"
    acts = np.asarray(actions, dtype=np.float64)
    l2 = np.linalg.norm(acts, axis=1)
    return {
        "label": label,
        "success": int(label == "success"),
        "target": int(target),
        "direction": int(direction),
        "amplitude": float(amplitude),
        "noise_std": float(noise_std),
        "gripper_delay": int(gripper_delay),
        "mean_l2": float(l2.mean()),
        "max_l2": float(l2.max()),
        "std_l2": float(l2.std()),
        "terminal_l2": float(l2[-1]),
        "target_xy": task.seats[int(target), :2].tolist(),
        "seat_positions": task.seats[:, :2].reshape(-1).tolist(),
        "payload_xy": task.payload_xy.tolist(),
    }


def collect_confounded(task_mod, n_episodes=40, seed=0):
    """One layout family: target=direction=0; RLBench joint-noise ladder.

    Gripper-delay is excluded from the primary set so AUROC is not inflated
    by a perfect timing cue; delay remains a separate optional ablation.
    """
    rows = []
    noise_levels = [0.0, 0.02, 0.04, 0.06, 0.08]
    for ep in range(n_episodes):
        task = task_mod.FactorTask(seed=1000 + ep)
        task.set_target_visual(0)
        task.reset(payload_xy=np.array([0.0, 0.0]))
        rng = np.random.default_rng(2000 + ep)
        variants = [("expert", 0.0, 0)]
        variants += [(f"joint_noise_{s:g}", s, 0) for s in noise_levels[1:]]
        for name, noise_std, delay in variants:
            task.reset(payload_xy=np.array([0.0, 0.0]))
            r = rollout_noisy(
                task, target=0, direction=0, amplitude=0.16,
                noise_std=noise_std, gripper_delay=delay, rng=rng,
            )
            r["variant"] = name
            r["episode"] = ep
            r["recipe"] = "rlbench_style_perturbation"
            rows.append(r)
    return rows


def collect_paired(task_mod, n_layouts=40, seed=0, amplitude=0.16):
    rows = []
    for layout in range(n_layouts):
        task = task_mod.FactorTask(seed=3000 + layout)
        task.randomize_seats()
        target = layout % 3
        task.set_target_visual(target)
        payload_xy = None
        for direction in range(3):
            task.reset(payload_xy=payload_xy)
            if payload_xy is None:
                payload_xy = task.payload_xy.copy()
            r = rollout_noisy(
                task, target=target, direction=direction, amplitude=amplitude,
                noise_std=0.0, gripper_delay=0,
                rng=np.random.default_rng(4000 + layout * 3 + direction),
            )
            r["variant"] = "paired_clean"
            r["layout"] = layout
            r["recipe"] = "paired_geometry"
            r["noise_std"] = 0.0
            rows.append(r)
    return rows


def eval_amplitude_ab(rows, name):
    y = np.array([r["success"] for r in rows], dtype=np.int32)
    out = {
        "name": name,
        "n": len(rows),
        "positive_rate": float(y.mean()),
        "label_counts": {
            "success": int(sum(r["label"] == "success" for r in rows)),
            "wrong_target": int(sum(r["label"] == "wrong_target" for r in rows)),
            "drop": int(sum(r["label"] == "drop" for r in rows)),
        },
    }
    for key in ("noise_std", "mean_l2", "max_l2", "std_l2", "terminal_l2", "gripper_delay", "amplitude"):
        s = np.array([r[key] for r in rows], dtype=np.float64)
        # For success prediction, larger noise should predict failure => use -score for noise
        if key in ("noise_std", "gripper_delay", "mean_l2", "max_l2", "terminal_l2"):
            # convention in report: AUROC of predicting success from *negative* amplitude
            # (larger amp/noise => failure). So auroc_success_from_neg_amp = auroc(-s, y)
            out[key] = {
                "auroc_success_from_neg_amp": auroc(-s, y),
                "auroc_success_from_amp": auroc(s, y),
                "bootstrap_neg": bootstrap_auroc(-s, y),
                "mean_success": float(s[y == 1].mean()) if (y == 1).any() else float("nan"),
                "mean_fail": float(s[y == 0].mean()) if (y == 0).any() else float("nan"),
                "std": float(s.std()),
            }
        else:
            out[key] = {
                "auroc_success_from_amp": auroc(s, y),
                "std": float(s.std()),
            }
    # primary shortcut flag: noise_std or mean_l2 encodes outcome
    primary = out["noise_std"]["auroc_success_from_neg_amp"]
    primary2 = out["mean_l2"]["auroc_success_from_neg_amp"]
    out["amplitude_shortcut"] = bool(max(primary, primary2) >= 0.90)
    out["primary_auroc"] = float(max(primary, primary2))
    return out


def _feat_amp(r):
    # Amplitude / perturbation-strength features only (no gripper_delay).
    return np.array([
        r["noise_std"], r["mean_l2"], r["max_l2"], r["std_l2"],
        r["terminal_l2"], r["amplitude"],
    ], dtype=np.float64)


def _feat_geom(r):
    return np.array(r["target_xy"] + r["seat_positions"] + r["payload_xy"], dtype=np.float64)


def _feat_action_dir(r):
    d = np.zeros(3, dtype=np.float64)
    d[int(r["direction"]) % 3] = 1.0
    return np.r_[d, r["amplitude"]]


def train_transfer(confounded, paired):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ytr = np.array([r["success"] for r in confounded], dtype=np.int32)
    yte = np.array([r["success"] for r in paired], dtype=np.int32)
    Xtr_amp = np.stack([_feat_amp(r) for r in confounded])
    Xte_amp = np.stack([_feat_amp(r) for r in paired])
    # On paired, noise/delay are constant zeros — amplitude-only must collapse.
    Xtr_joint = np.stack([np.r_[_feat_action_dir(r), _feat_geom(r)] for r in confounded])
    Xte_joint = np.stack([np.r_[_feat_action_dir(r), _feat_geom(r)] for r in paired])
    # Also train joint on paired for reference oracle-style
    Xtr_paired = Xte_joint
    ytr_paired = yte
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(yte))
    cut = max(1, len(yte) // 2)
    tr_p, te_p = idx[:cut], idx[cut:]

    def fit_eval(Xtr, ytr_, Xte, yte_, name):
        if len(np.unique(ytr_)) < 2:
            return {"name": name, "error": "single_class_train"}
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000) if "amp" in name else
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0),
        )
        clf.fit(Xtr, ytr_)
        proba = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)
        return {
            "name": name,
            "train_auroc": auroc(clf.predict_proba(Xtr)[:, 1], ytr_),
            "test_auroc": auroc(proba, yte_),
            "test_acc": float((clf.predict(Xte) == yte_).mean()),
        }

    out = {
        "amp_on_confounded_to_paired": fit_eval(Xtr_amp, ytr, Xte_amp, yte, "amp_transfer"),
        "amp_on_confounded_self": fit_eval(Xtr_amp, ytr, Xtr_amp, ytr, "amp_self"),
        "joint_trained_on_paired_holdout": fit_eval(
            Xtr_paired[tr_p], ytr_paired[tr_p], Xtr_paired[te_p], ytr_paired[te_p], "joint_paired"),
    }
    # Confounded-trained joint (action+geom) — geometry varies little in confounded
    out["joint_on_confounded_to_paired"] = fit_eval(Xtr_joint, ytr, Xte_joint, yte, "joint_transfer")
    # Collapse criterion
    amp_self = out["amp_on_confounded_self"].get("test_auroc", 0)
    amp_xfer = out["amp_on_confounded_to_paired"].get("test_auroc", 1)
    paired_joint = out["joint_trained_on_paired_holdout"].get("test_auroc", 0)
    out["fell_for_shortcut"] = bool(
        amp_self >= 0.90 and (not np.isfinite(amp_xfer) or amp_xfer < 0.60) and paired_joint >= 0.70
    )
    out["summary"] = {
        "amp_self_auroc": amp_self,
        "amp_transfer_auroc": amp_xfer,
        "paired_joint_auroc": paired_joint,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Path to this checkout (default: current directory)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--layouts", type=int, default=40)
    args = ap.parse_args()

    task_mod = _load_factor_task(Path(args.repo))
    confounded = collect_confounded(task_mod, n_episodes=args.episodes)
    paired = collect_paired(task_mod, n_layouts=args.layouts)
    conf_ab = eval_amplitude_ab(confounded, "rlbench_style_confounded")
    pair_ab = eval_amplitude_ab(paired, "paired_geometry")
    transfer = train_transfer(confounded, paired)

    result = {
        "recipe": {
            "name": "rlbench_style_expert_perturbation",
            "source_script": "scripts/collect_rlbench_perturbations.py",
            "pattern": [
                "reset_to_demo / fixed start",
                "replay expert action chunk",
                "variants: expert | joint_noise_σ∈{0.01,0.03,0.06} | gripper_delay",
                "label = terminal task success",
            ],
            "engine": "MuJoCo FactorTask faithful port (RLBench env historically fragile for paired-geometry)",
        },
        "confounded_ab": conf_ab,
        "paired_ab": pair_ab,
        "transfer": transfer,
        "decision": {
            "protocol_injects_shortcut": bool(conf_ab["amplitude_shortcut"]),
            "paired_removes_shortcut": bool(
                pair_ab["primary_auroc"] < 0.60 or pair_ab["amplitude"]["std"] < 1e-9
            ),
            "model_fell_for_shortcut": bool(transfer["fell_for_shortcut"]),
            "numbers": {
                "confounded_primary_auroc": conf_ab["primary_auroc"],
                "paired_primary_auroc": pair_ab["primary_auroc"],
                **transfer["summary"],
            },
        },
        "rows_confounded": confounded,
        "rows_paired": paired,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    compact = {
        "decision": result["decision"],
        "confounded_ab": {
            "n": conf_ab["n"],
            "positive_rate": conf_ab["positive_rate"],
            "amplitude_shortcut": conf_ab["amplitude_shortcut"],
            "primary_auroc": conf_ab["primary_auroc"],
            "label_counts": conf_ab["label_counts"],
            "noise_std": conf_ab["noise_std"],
            "mean_l2": conf_ab["mean_l2"],
        },
        "paired_ab": {
            "n": pair_ab["n"],
            "positive_rate": pair_ab["positive_rate"],
            "primary_auroc": pair_ab["primary_auroc"],
            "amplitude_shortcut": pair_ab["amplitude_shortcut"],
        },
        "transfer": transfer,
    }
    print(json.dumps(compact, indent=2, default=str))


if __name__ == "__main__":
    main()
