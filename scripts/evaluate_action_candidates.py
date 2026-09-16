"""Evaluate action-risk ranking on clips with multiple executed candidates."""

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
    ap.add_argument("--data", default="data/features/candidates_open_drawer/test")
    ap.add_argument("--checkpoint", default="results/action4d_candidates_open_drawer.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ClipDataset(args.data)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = state["model_config"]
    model = ActionConditioned4DModel(
        robot_state_dim=cfg["robot_state_dim"], action_dim=cfg["action_dim"],
        future_steps=cfg["future_steps"], hidden_dim=cfg["hidden_dim"],
        layers=cfg["layers"], heads=cfg["heads"], point_feature_dim=cfg["point_feature_dim"],
        relative_actions=bool(cfg.get("relative_actions", False)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    predicted, labels = [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
            out = model(item["points"], item["actions"], item["robot_state"], item["features"])
            predicted.append(torch.sigmoid(out.collision_logits).squeeze(0).cpu())
            labels.append(item["collision"].squeeze(0).cpu())
    risk, y = torch.stack(predicted), torch.stack(labels)
    select = risk.argmin(dim=1)
    random_select = torch.randint(risk.shape[1], (len(ds),))
    oracle = y.argmin(dim=1)
    chosen = lambda index: y.gather(1, index[:, None]).squeeze(1)
    flat_risk, flat_y = risk.flatten(), y.flatten()
    order = torch.argsort(flat_risk, descending=True)
    if flat_y.max() > 0 and flat_y.min() < 1:
        pos, neg = flat_y.sum(), (1 - flat_y).sum()
        auroc = float(torch.trapz(torch.cumsum(flat_y[order], 0) / pos,
                                  torch.cumsum(1 - flat_y[order], 0) / neg))
    else:
        auroc = float("nan")
    print({
        "clips": len(ds), "candidates": int(risk.shape[1]), "collision_auroc": auroc,
        "fixed_expert_collision_rate": float(y[:, 0].mean()),
        "random_collision_rate": float(chosen(random_select).mean()),
        "risk_driven_collision_rate": float(chosen(select).mean()),
        "oracle_collision_rate": float(chosen(oracle).mean()),
        "risk_driven_oracle_regret": float((chosen(select) - chosen(oracle)).mean()),
    })


if __name__ == "__main__":
    main()
