"""Paper-facing evaluation for candidate risk and utility prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel
from scripts.train_dataset import ClipDataset


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean()
                  + 0.5 * (pos[:, None] == neg[None]).mean()))


def ece(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            value += mask.mean() * abs(prob[mask].mean() - label[mask].mean())
    return float(value)


def bootstrap_delta(a: np.ndarray, b: np.ndarray, seed: int, draws: int = 5000):
    rng = np.random.default_rng(seed)
    delta = a - b
    samples = rng.choice(delta, (draws, len(delta)), replace=True).mean(1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load_model(checkpoint: str, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = state["model_config"]
    # Older checkpoints trained with relative actions omitted this flag. Infer it
    # from the first action encoder weight so those results remain reproducible.
    relative_actions = bool(cfg.get("relative_actions", False))
    weight = state.get("model", {}).get("action_encoder.0.weight")
    if weight is not None and weight.shape[1] == 2 * int(cfg["action_dim"]):
        relative_actions = True
    model = ActionConditioned4DModel(
        robot_state_dim=cfg["robot_state_dim"], action_dim=cfg["action_dim"],
        future_steps=cfg["future_steps"], hidden_dim=cfg["hidden_dim"],
        layers=cfg["layers"], heads=cfg["heads"],
        point_feature_dim=cfg["point_feature_dim"],
        relative_actions=relative_actions,
    ).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--collision-weight", type=float, default=1.0)
    ap.add_argument("--failure-weight", type=float, default=1.0)
    ap.add_argument("--calibration", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ClipDataset(args.data)
    model = load_model(args.checkpoint, device)
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    rng = np.random.default_rng(args.seed)
    rows = []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = ds[i]
            m = item["actions"].shape[0]
            perm = torch.as_tensor(rng.permutation(m), dtype=torch.long)
            nominal_index = int(torch.where(perm == 0)[0].item())
            candidate_keys = {"actions", "future_delta", "affordance", "success", "collision"}
            item = {k: (v[perm] if k in candidate_keys else v) for k, v in item.items()}
            batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
            out = model(batch["points"], batch["actions"], batch["robot_state"], batch["features"])
            success_logits = out.success_logits[0]
            collision_logits = out.collision_logits[0]
            if calibration:
                success_logits = success_logits / float(calibration["success_temperature"])
                collision_logits = collision_logits / float(calibration["collision_temperature"])
            success_prob = torch.sigmoid(success_logits).cpu().numpy()
            collision_prob = torch.sigmoid(collision_logits).cpu().numpy()
            success = item["success"].numpy()
            collision = item["collision"].numpy()
            actions = item["actions"].numpy()
            true_utility = args.failure_weight * (1.0 - success) + args.collision_weight * collision
            failure_prob = 1.0 - success_prob
            if calibration:
                selection_failure = np.clip(
                    failure_prob + float(calibration["failure_residual_quantile"]), 0, 1)
                selection_collision = np.clip(
                    collision_prob + float(calibration["collision_residual_quantile"]), 0, 1)
            else:
                selection_failure, selection_collision = failure_prob, collision_prob
            predicted = (args.failure_weight * selection_failure
                         + args.collision_weight * selection_collision)
            rows.append({"success_prob": success_prob, "collision_prob": collision_prob,
                         "success": success, "collision": collision,
                         "true_utility": true_utility, "predicted": predicted,
                         "action_l2": np.linalg.norm(actions, axis=1),
                         "nominal_index": nominal_index})

    stack = lambda key: np.stack([row[key] for row in rows])
    sp, cp = stack("success_prob"), stack("collision_prob")
    s, c = stack("success"), stack("collision")
    true, pred, norm = stack("true_utility"), stack("predicted"), stack("action_l2")
    n, m = true.shape
    idx = np.arange(n)
    nominal = np.asarray([row["nominal_index"] for row in rows], dtype=np.int64)
    selections = {
        "fixed_nominal": nominal,
        "random_expected": None,
        "min_action_l2": norm.argmin(1),
        "max_action_l2": norm.argmax(1),
        "risk": pred.argmin(1),
        "oracle": true.argmin(1),
    }
    summary = {}
    per_clip_utility = {}
    for name, select in selections.items():
        if select is None:
            utility = true.mean(1); success = s.mean(1); collision = c.mean(1)
        else:
            utility = true[idx, select]; success = s[idx, select]; collision = c[idx, select]
        per_clip_utility[name] = utility
        summary[name] = {"utility": float(utility.mean()),
                         "success": float(success.mean()),
                         "collision": float(collision.mean())}

    flat_s, flat_c = s.reshape(-1), c.reshape(-1)
    flat_sp, flat_cp = sp.reshape(-1), cp.reshape(-1)
    coverage = {}
    order = np.argsort(pred.reshape(-1))
    flat_true = true.reshape(-1)
    for fraction in (0.2, 0.5, 0.8, 1.0):
        keep = order[:max(1, int(len(order) * fraction))]
        coverage[str(fraction)] = float(flat_true[keep].mean())

    result = {
        "clips": n, "candidates": m, "candidate_order_permuted": True,
        "metrics": {
            "failure_auroc": auroc(1.0 - flat_sp, 1.0 - flat_s),
            "collision_auroc": auroc(flat_cp, flat_c),
            "success_brier": float(np.mean((flat_sp - flat_s) ** 2)),
            "collision_brier": float(np.mean((flat_cp - flat_c) ** 2)),
            "success_ece": ece(flat_sp, flat_s),
            "collision_ece": ece(flat_cp, flat_c),
        },
        "selection": summary,
        "risk_minus_fixed_utility_ci95": bootstrap_delta(
            per_clip_utility["risk"], per_clip_utility["fixed_nominal"],
            args.seed, args.bootstrap),
        "risk_coverage_utility": coverage,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
