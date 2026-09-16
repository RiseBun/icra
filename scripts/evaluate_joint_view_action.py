"""Evaluate joint active-view and action-risk selection on synchronized clips."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel
from scripts.train_dataset import ClipDataset


CAMERAS = ("front", "left_shoulder", "right_shoulder", "overhead", "wrist")


def load_model(checkpoint, sample, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = state["model_config"]
    model = ActionConditioned4DModel(
        robot_state_dim=int(cfg["robot_state_dim"]), action_dim=int(cfg["action_dim"]),
        future_steps=int(cfg["future_steps"]), hidden_dim=int(cfg["hidden_dim"]),
        layers=int(cfg["layers"]), heads=int(cfg["heads"]),
        point_feature_dim=int(cfg["point_feature_dim"]),
    ).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def predict_dataset(ds, model, device):
    risks, labels = [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
            output = model(item["points"], item["actions"], item["robot_state"], item["features"])
            risks.append(torch.sigmoid(output.collision_logits).squeeze(0).cpu())
            labels.append(item["collision"].squeeze(0).cpu())
    return torch.stack(risks), torch.stack(labels)


def selected_rate(labels, view_index, action_index):
    n = torch.arange(labels.shape[0])
    return labels[n, view_index, action_index].float().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features/candidates_joint_test")
    ap.add_argument("--checkpoint", default="results/action4d_candidates_open_drawer_strict.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [ClipDataset(str(Path(args.data) / camera)) for camera in CAMERAS]
    if len({len(ds) for ds in datasets}) != 1:
        raise ValueError("all camera datasets must have the same number of clips")
    model = load_model(args.checkpoint, datasets[0][0], device)
    predictions = [predict_dataset(ds, model, device) for ds in datasets]
    risks = torch.stack([item[0] for item in predictions], dim=1)  # [N,V,M]
    labels = torch.stack([item[1] for item in predictions], dim=1)  # [N,V,M]
    if not torch.equal(labels, labels[:, :1].expand_as(labels)):
        raise ValueError("candidate labels disagree across synchronized views")

    n = risks.shape[0]
    front_view = torch.zeros(n, dtype=torch.long)
    random_view = torch.randint(len(CAMERAS), (n,))
    risk_view = risks.min(dim=2).values.argmin(dim=1)
    oracle_view = labels.min(dim=2).values.argmin(dim=1)
    fixed_action = risks[:, 0].argmin(dim=1)
    random_action = risks[torch.arange(n), random_view].argmin(dim=1)
    risk_action = risks[torch.arange(n), risk_view].argmin(dim=1)
    oracle_action = labels[torch.arange(n), oracle_view].argmin(dim=1)
    out = {
        "clips": int(n), "views": len(CAMERAS), "candidates": int(risks.shape[2]),
        "fixed_front_collision_rate": float(selected_rate(labels, front_view, fixed_action)),
        "random_view_collision_rate": float(selected_rate(labels, random_view, random_action)),
        "risk_driven_view_action_collision_rate": float(selected_rate(labels, risk_view, risk_action)),
        "oracle_view_action_collision_rate": float(selected_rate(labels, oracle_view, oracle_action)),
        "risk_view_selected_fraction": [float((risk_view == i).float().mean()) for i in range(len(CAMERAS))],
        "risk_action_oracle_regret": float(
            selected_rate(labels, risk_view, risk_action) - selected_rate(labels, oracle_view, oracle_action)
        ),
        "mean_min_predicted_risk_front": float(risks[:, 0].min(dim=1).values.mean()),
        "mean_min_predicted_risk_joint": float(risks.min(dim=2).values.min(dim=1).values.mean()),
    }
    print(out)


if __name__ == "__main__":
    main()
