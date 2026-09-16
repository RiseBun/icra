"""Sweep success gates for risk-aware candidate selection."""

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
    ap.add_argument("--gates", type=float, nargs="+", default=(0.1, 0.3, 0.5, 0.7, 0.9))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ClipDataset(args.data)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = state["model_config"]
    model = ActionConditioned4DModel(
        robot_state_dim=cfg["robot_state_dim"], action_dim=cfg["action_dim"],
        future_steps=cfg["future_steps"], hidden_dim=cfg["hidden_dim"],
        layers=cfg["layers"], heads=cfg["heads"], point_feature_dim=cfg["point_feature_dim"],
        relative_actions=bool(cfg.get("relative_actions", False)),
    ).to(device); model.load_state_dict(state["model"]); model.eval()
    probs, risk, success, collision = [], [], [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
            out = model(item["points"], item["actions"], item["robot_state"], item["features"])
            probs.append(torch.sigmoid(out.success_logits).squeeze(0).cpu())
            risk.append(torch.sigmoid(out.collision_logits).squeeze(0).cpu())
            success.append(item["success"].squeeze(0).cpu()); collision.append(item["collision"].squeeze(0).cpu())
    p, r, y, c = map(torch.stack, (probs, risk, success, collision))
    n = torch.arange(len(ds))
    for gate in args.gates:
        allowed = p >= gate
        masked = r.masked_fill(~allowed, float("inf"))
        select = masked.argmin(1)
        fallback = torch.zeros(len(ds), dtype=torch.long)
        select = torch.where(torch.isfinite(masked.gather(1, select[:, None]).squeeze(1)), select, fallback)
        print({"gate": gate, "selected_success": float(y[n, select].mean()),
               "selected_collision": float(c[n, select].mean()),
               "fallback_fraction": float((~allowed.any(1)).float().mean()),
               "utility": float((1.0-y[n,select]+c[n,select]).mean())})


if __name__ == "__main__":
    main()
