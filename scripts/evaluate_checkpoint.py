"""Evaluate risk calibration and future-motion error on feature-contract clips."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel
from scripts.train_dataset import ClipDataset


def ece(prob, label, bins=10):
    edges = torch.linspace(0, 1, bins + 1, device=prob.device)
    value = prob.new_zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            value = value + mask.float().mean() * (prob[mask].mean() - label[mask].mean()).abs()
    return value


def auroc(score, label):
    order = torch.argsort(score, descending=True)
    y = label[order]
    positives = y.sum().clamp_min(1.0)
    negatives = (1.0 - y).sum().clamp_min(1.0)
    tpr = torch.cumsum(y, 0) / positives
    fpr = torch.cumsum(1.0 - y, 0) / negatives
    return torch.trapz(tpr, fpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features/synthetic")
    ap.add_argument("--checkpoint", default="results/action4d_synthetic.pt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ClipDataset(args.data)
    probe = ds[0]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state.get("model_config", {})
    model = ActionConditioned4DModel(
        robot_state_dim=int(config.get("robot_state_dim", probe["robot_state"].numel())),
        action_dim=int(config.get("action_dim", probe["actions"].shape[-1])),
        future_steps=int(config.get("future_steps", probe["future_delta"].shape[1])),
        hidden_dim=int(config.get("hidden_dim", 256)),
        layers=int(config.get("layers", 4)),
        heads=int(config.get("heads", 8)),
        point_feature_dim=int(config.get("point_feature_dim", probe["features"].shape[-1])),
        relative_actions=bool(config.get("relative_actions", False)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    probs, labels, risks, targets, preds = [], [], [], [], []
    uncertainties, flow_errors, flow_mae = [], [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
            out = model(item["points"], item["actions"], item["robot_state"], item["features"])
            probs.append(torch.sigmoid(out.success_logits).flatten())
            labels.append(item["success"].flatten())
            risks.append(torch.sigmoid(out.collision_logits).flatten())
            targets.append(item["collision"].flatten())
            preds.append(torch.nn.functional.smooth_l1_loss(out.future_delta, item["future_delta"]))
            uncertainties.append(out.uncertainty.flatten())
            flow_errors.append(torch.nn.functional.smooth_l1_loss(
                out.future_delta, item["future_delta"], reduction="none"
            ).mean(dim=(2, 3, 4)).flatten())
            flow_mae.append(torch.abs(out.future_delta - item["future_delta"]).mean(dim=(2, 3, 4)).flatten())
    p, y = torch.cat(probs), torch.cat(labels)
    r, z = torch.cat(risks), torch.cat(targets)
    brier = ((p - y) ** 2).mean()
    collision_brier = ((r - z) ** 2).mean()
    collision_auroc = auroc(r, z) if z.max() > 0 and z.min() < 1 else torch.nan
    risk_order = torch.argsort(r)
    coverage = max(1, int(0.8 * len(risk_order)))
    selective_collision = z[risk_order[:coverage]].mean()
    u, f = torch.cat(uncertainties), torch.cat(flow_errors)
    mae = torch.cat(flow_mae)
    u0, f0 = u - u.mean(), f - f.mean()
    uncertainty_error_corr = (u0 * f0).mean() / (u0.std() * f0.std() + 1e-8)
    u_order = torch.argsort(u)
    selective_flow = f[u_order[:coverage]].mean()
    print({
        "clips": len(ds),
        "success_brier": float(brier),
        "success_ece": float(ece(p, y)),
        "collision_brier": float(collision_brier),
        "collision_ece": float(ece(r, z)),
        "collision_auroc": float(collision_auroc),
        "collision_rate_lowest_risk_80pct": float(selective_collision),
        "future_smooth_l1": float(torch.stack(preds).mean()),
        "future_mae": float(mae.mean()),
        "uncertainty_error_corr": float(uncertainty_error_corr),
        "future_error_lowest_uncertainty_80pct": float(selective_flow),
    })


if __name__ == "__main__":
    main()
