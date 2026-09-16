"""Evaluate fixed, confidence, random, risk-driven, and oracle view selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel
from scripts.train_dataset import ClipDataset


def auroc(score, label):
    if label.max() <= 0 or label.min() >= 1:
        return float("nan")
    order = torch.argsort(score, descending=True)
    y = label[order]
    pos = y.sum()
    neg = (1 - y).sum()
    return float(torch.trapz(torch.cumsum(y, 0) / pos, torch.cumsum(1 - y, 0) / neg))


def load_model(checkpoint, sample, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = state.get("model_config", {})
    model = ActionConditioned4DModel(
        robot_state_dim=int(cfg.get("robot_state_dim", sample["robot_state"].numel())),
        action_dim=int(cfg.get("action_dim", sample["actions"].shape[-1])),
        future_steps=int(cfg.get("future_steps", sample["future_delta"].shape[1])),
        hidden_dim=int(cfg.get("hidden_dim", 256)), layers=int(cfg.get("layers", 4)),
        heads=int(cfg.get("heads", 8)), point_feature_dim=int(cfg.get("point_feature_dim", 1)),
        relative_actions=bool(cfg.get("relative_actions", False)),
    ).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def predict(root, checkpoint, device):
    ds = ClipDataset(root)
    model = load_model(checkpoint, ds[0], device)
    risks, labels, confidence = [], [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
            out = model(item["points"], item["actions"], item["robot_state"], item["features"])
            risks.append(torch.sigmoid(out.collision_logits).mean(dim=1).squeeze(0).cpu())
            labels.append(item["collision"].flatten()[0].cpu())
            confidence.append(item["features"].mean(dim=(1, 2, 3)).squeeze(0).cpu())
    return torch.stack(risks), torch.stack(labels), torch.stack(confidence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features/multiview_eval")
    ap.add_argument("--checkpoint", default="results/action4d_omega_conf_weighted.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cameras = ("front", "left_shoulder", "right_shoulder", "overhead", "wrist")
    results = {c: predict(str(Path(args.data) / c), args.checkpoint, device) for c in cameras}
    risks = torch.stack([results[c][0] for c in cameras], dim=1)
    labels = torch.stack([results[c][1] for c in cameras], dim=1)
    confidence = torch.stack([results[c][2] for c in cameras], dim=1)
    # Labels should be view-invariant; use the majority value and report disagreement.
    y = (labels.mean(dim=1) >= 0.5).float()
    selected = {
        "fixed_front": torch.zeros(len(y), dtype=torch.long),
        "random": torch.randint(0, len(cameras), (len(y),)),
        "max_confidence": confidence.argmax(dim=1),
        "risk_driven": risks.argmin(dim=1),
        "oracle": labels.argmin(dim=1),
    }
    out = {"clips": int(len(y)), "label_disagreement": float((labels != labels[:, :1]).any(dim=1).float().mean())}
    for name, index in selected.items():
        chosen_risk = risks.gather(1, index[:, None]).squeeze(1)
        chosen_label = labels.gather(1, index[:, None]).squeeze(1)
        out[name] = {
            "collision_rate": float(chosen_label.mean()),
            "mean_predicted_risk": float(chosen_risk.mean()),
            "risk_brier": float(((chosen_risk - chosen_label) ** 2).mean()),
        }
    out["per_view_auroc"] = {
        camera: auroc(risks[:, i], y) for i, camera in enumerate(cameras)
    }
    print(out)


if __name__ == "__main__":
    main()
