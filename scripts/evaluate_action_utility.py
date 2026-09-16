"""Evaluate joint safety/task utility for candidate action selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel
from scripts.train_dataset import ClipDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--collision-weight", type=float, default=1.0)
    ap.add_argument("--failure-weight", type=float, default=1.0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ClipDataset(args.data)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = state["model_config"]
    relative_actions = bool(cfg.get("relative_actions", False))
    weight = state.get("model", {}).get("action_encoder.0.weight")
    if weight is not None and weight.shape[1] == 2 * int(cfg["action_dim"]):
        relative_actions = True
    model = ActionConditioned4DModel(
        robot_state_dim=cfg["robot_state_dim"], action_dim=cfg["action_dim"],
        future_steps=cfg["future_steps"], hidden_dim=cfg["hidden_dim"],
        layers=cfg["layers"], heads=cfg["heads"], point_feature_dim=cfg["point_feature_dim"],
        relative_actions=relative_actions,
    ).to(device)
    model.load_state_dict(state["model"]); model.eval()
    risks, failures, collisions = [], [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
            out = model(item["points"], item["actions"], item["robot_state"], item["features"])
            risks.append(torch.sigmoid(out.collision_logits).squeeze(0).cpu())
            failures.append((1.0 - item["success"].squeeze(0)).cpu())
            collisions.append(item["collision"].squeeze(0).cpu())
    r, f, c = torch.stack(risks), torch.stack(failures), torch.stack(collisions)
    # The success logit is encoded in the same model output, so re-run to retain it.
    success_probs = []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
            out = model(item["points"], item["actions"], item["robot_state"], item["features"])
            success_probs.append(torch.sigmoid(out.success_logits).squeeze(0).cpu())
    sp = torch.stack(success_probs)
    predicted_utility = args.collision_weight * r + args.failure_weight * (1.0 - sp)
    true_utility = args.collision_weight * c + args.failure_weight * f
    n = torch.arange(len(ds))
    select = predicted_utility.argmin(1)
    random_select = torch.randint(r.shape[1], (len(ds),))
    oracle = true_utility.argmin(1)
    chosen = lambda x, v: x[n, v].mean()
    print({
        "clips": len(ds), "candidates": int(r.shape[1]),
        "fixed_expert_utility": float(true_utility[:, 0].mean()),
        "random_utility": float(chosen(true_utility, random_select)),
        "risk_utility": float(chosen(true_utility, select)),
        "oracle_utility": float(chosen(true_utility, oracle)),
        "risk_collision": float(chosen(c, select)),
        "risk_failure": float(chosen(f, select)),
        "risk_selection": select.tolist(),
    })


if __name__ == "__main__":
    main()
