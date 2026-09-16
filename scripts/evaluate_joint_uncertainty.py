"""Sweep uncertainty-penalized joint view/action selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel
from scripts.train_dataset import ClipDataset


CAMERAS = ("front", "left_shoulder", "right_shoulder", "overhead", "wrist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--betas", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [ClipDataset(str(Path(args.data) / camera)) for camera in CAMERAS]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = state["model_config"]
    model = ActionConditioned4DModel(
        robot_state_dim=cfg["robot_state_dim"], action_dim=cfg["action_dim"],
        future_steps=cfg["future_steps"], hidden_dim=cfg["hidden_dim"],
        layers=cfg["layers"], heads=cfg["heads"], point_feature_dim=cfg["point_feature_dim"],
        relative_actions=bool(cfg.get("relative_actions", False)),
    ).to(device); model.load_state_dict(state["model"]); model.eval()
    all_risk, all_unc, labels = [], [], []
    with torch.inference_mode():
        for ds in datasets:
            rr, uu, yy = [], [], []
            for i in range(len(ds)):
                item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
                out = model(item["points"], item["actions"], item["robot_state"], item["features"])
                rr.append(torch.sigmoid(out.collision_logits).squeeze(0).cpu())
                uu.append(out.uncertainty.squeeze(0).cpu())
                yy.append(item["collision"].squeeze(0).cpu())
            all_risk.append(torch.stack(rr)); all_unc.append(torch.stack(uu)); labels.append(torch.stack(yy))
    risk, unc, y = torch.stack(all_risk, 1), torch.stack(all_unc, 1), torch.stack(labels, 1)
    n = torch.arange(risk.shape[0])
    for beta in args.betas:
        score = risk + beta * unc
        view, action = score.flatten(1).min(1), score.flatten(1).argmin(1)
        vi, ai = action // risk.shape[2], action % risk.shape[2]
        chosen = y[n, vi, ai]
        print({"beta": beta, "collision_rate": float(chosen.mean()),
               "mean_predicted_risk": float(risk[n, vi, ai].mean()),
               "mean_uncertainty": float(unc[n, vi, ai].mean()),
               "view_fraction": [float((vi == i).float().mean()) for i in range(len(CAMERAS))]})


if __name__ == "__main__":
    main()
