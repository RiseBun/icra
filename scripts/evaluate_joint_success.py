"""Evaluate joint view/action selection using success probability."""

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
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [ClipDataset(str(Path(args.data) / camera)) for camera in CAMERAS]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True); cfg = state["model_config"]
    model = ActionConditioned4DModel(
        robot_state_dim=cfg["robot_state_dim"], action_dim=cfg["action_dim"],
        future_steps=cfg["future_steps"], hidden_dim=cfg["hidden_dim"],
        layers=cfg["layers"], heads=cfg["heads"], point_feature_dim=cfg["point_feature_dim"],
        relative_actions=bool(cfg.get("relative_actions", False)),
    ).to(device); model.load_state_dict(state["model"]); model.eval()
    probs, labels = [], []
    with torch.inference_mode():
        for ds in datasets:
            pp, yy = [], []
            for i in range(len(ds)):
                item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
                out = model(item["points"], item["actions"], item["robot_state"], item["features"])
                pp.append(torch.sigmoid(out.success_logits).squeeze(0).cpu())
                yy.append(item["success"].squeeze(0).cpu())
            probs.append(torch.stack(pp)); labels.append(torch.stack(yy))
    p, y = torch.stack(probs, 1), torch.stack(labels, 1)
    if not torch.equal(y, y[:, :1].expand_as(y)):
        raise ValueError("success labels disagree across views")
    n = torch.arange(len(y))
    front_action = p[:, 0].argmax(1)
    random_view = torch.randint(len(CAMERAS), (len(y),)); random_action = p[n, random_view].argmax(1)
    joint_view = p.max(2).values.argmax(1); joint_action = p[n, joint_view].argmax(1)
    oracle_view = y.max(2).values.argmax(1); oracle_action = y[n, oracle_view].argmax(1)
    chosen = lambda vi, ai: y[n, vi, ai]
    print({
        "clips": len(y), "views": len(CAMERAS), "candidates": int(p.shape[2]),
        "fixed_front_success": float(chosen(torch.zeros(len(y), dtype=torch.long), front_action).mean()),
        "random_view_success": float(chosen(random_view, random_action).mean()),
        "joint_success": float(chosen(joint_view, joint_action).mean()),
        "oracle_success": float(chosen(oracle_view, oracle_action).mean()),
        "joint_view_fraction": [float((joint_view == i).float().mean()) for i in range(len(CAMERAS))],
    })


if __name__ == "__main__":
    main()
