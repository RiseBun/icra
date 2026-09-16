"""Evaluate v2 geometry, spatial risk, calibration and candidate selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import RiskCalibration, load_risk4d_checkpoint


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    positive, negative = score[label > 0.5], score[label <= 0.5]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    return float((positive[:, None] > negative[None]).mean()
                 + 0.5 * (positive[:, None] == negative[None]).mean())


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    label = label > 0.5
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score)
    sorted_label = label[order]
    precision = np.cumsum(sorted_label) / (np.arange(len(label)) + 1)
    return float(precision[sorted_label].sum() / positives)


def ece(probability: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability < edges[index + 1] if index + 1 < bins else probability <= 1.0)
        if mask.any():
            total += mask.mean() * abs(probability[mask].mean() - label[mask].mean())
    return float(total)


def binary_iou(probability: np.ndarray, label: np.ndarray) -> float:
    prediction = probability >= 0.5
    target = label >= 0.5
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else float("nan")


def bootstrap_ci(delta: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, (draws, len(delta)), replace=True).mean(1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--failure-weight", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_risk4d_checkpoint(path, device) for path in args.checkpoints]
    calibration = RiskCalibration.load(args.calibration) if args.calibration else RiskCalibration()
    dataset = Risk4DDataset(args.data)
    rows = []
    with torch.inference_mode():
        for item in dataset:
            batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
            outputs = [model(
                batch["points"], batch["point_features"], batch["robot_state"],
                batch["robot_action_flow"], batch["task_embedding"],
            ) for model in models]
            success_logits = torch.stack([output.success_logits for output in outputs]).mean(0)
            collision_logits = torch.stack([output.collision_logits for output in outputs]).mean(0)
            member_probability = torch.stack([
                torch.stack((output.success_logits.sigmoid(), output.collision_logits.sigmoid()), -1)
                for output in outputs
            ])
            epistemic = member_probability.var(0, correction=0).mean(-1)
            success_probability, collision_probability = calibration.probabilities(
                success_logits, collision_logits)
            risk = calibration.risk_bound(
                success_logits, collision_logits, epistemic,
                args.failure_weight, args.collision_weight,
            )
            rows.append({
                "flow": torch.stack([output.future_point_flow for output in outputs]).mean(0).cpu().numpy()[0],
                "contact": torch.stack([output.task_contact_logits.sigmoid() for output in outputs]).mean(0).cpu().numpy()[0],
                "spatial_collision": torch.stack([
                    output.harmful_collision_logits.sigmoid() for output in outputs
                ]).mean(0).cpu().numpy()[0],
                "success_probability": success_probability.cpu().numpy()[0],
                "collision_probability": collision_probability.cpu().numpy()[0],
                "risk": risk.cpu().numpy()[0],
                "future_point_flow": item["future_point_flow"].numpy(),
                "task_contact_map": item["task_contact_map"].numpy(),
                "harmful_collision_map": item["harmful_collision_map"].numpy(),
                "success": item["success"].numpy(),
                "collision": item["collision"].numpy(),
            })

    stack = lambda key: np.stack([row[key] for row in rows])
    flow, flow_target = stack("flow"), stack("future_point_flow")
    contact, contact_target = stack("contact"), stack("task_contact_map")
    spatial_collision = stack("spatial_collision")
    spatial_collision_target = stack("harmful_collision_map")
    success_probability, success = stack("success_probability"), stack("success")
    collision_probability, collision = stack("collision_probability"), stack("collision")
    risk = stack("risk")
    flow_epe = np.linalg.norm(flow - flow_target, axis=-1)
    dynamic = np.linalg.norm(flow_target, axis=-1) > 0.01
    actual_risk = args.failure_weight * (1.0 - success) + args.collision_weight * collision
    sample_index = np.arange(len(dataset))
    first = np.zeros(len(dataset), dtype=np.int64)
    selected = risk.argmin(1)
    oracle = actual_risk.argmin(1)
    selected_success = success[sample_index, selected]
    first_success = success[sample_index, first]
    selected_collision = collision[sample_index, selected]
    first_collision = collision[sample_index, first]
    order = np.argsort(risk.reshape(-1))
    flat_actual = actual_risk.reshape(-1)
    coverage = {}
    for fraction in (0.2, 0.5, 0.8, 1.0):
        keep = order[:max(1, int(len(order) * fraction))]
        coverage[str(fraction)] = float(flat_actual[keep].mean())

    result = {
        "clips": len(dataset),
        "ensemble_members": len(models),
        "geometry": {
            "flow_epe": float(flow_epe.mean()),
            "dynamic_flow_epe": float(flow_epe[dynamic].mean()) if dynamic.any() else float("nan"),
        },
        "spatial_risk": {
            "contact_auprc": average_precision(contact.reshape(-1), contact_target.reshape(-1)),
            "contact_miou": binary_iou(contact, contact_target),
            "harmful_collision_auprc": average_precision(
                spatial_collision.reshape(-1), spatial_collision_target.reshape(-1)),
            "harmful_collision_miou": binary_iou(spatial_collision, spatial_collision_target),
        },
        "candidate_risk": {
            "failure_auroc": auroc(1.0 - success_probability.reshape(-1), 1.0 - success.reshape(-1)),
            "collision_auroc": auroc(collision_probability.reshape(-1), collision.reshape(-1)),
            "success_brier": float(np.mean((success_probability - success) ** 2)),
            "collision_brier": float(np.mean((collision_probability - collision) ** 2)),
            "success_ece": ece(success_probability.reshape(-1), success.reshape(-1)),
            "collision_ece": ece(collision_probability.reshape(-1), collision.reshape(-1)),
            "risk_coverage": coverage,
        },
        "selection": {
            "first_sample_success": float(first_success.mean()),
            "risk_selected_success": float(selected_success.mean()),
            "oracle_success": float(success[sample_index, oracle].mean()),
            "first_sample_collision": float(first_collision.mean()),
            "risk_selected_collision": float(selected_collision.mean()),
            "oracle_collision": float(collision[sample_index, oracle].mean()),
            "success_gain_ci95": bootstrap_ci(
                selected_success - first_success, args.seed, args.bootstrap),
            "collision_reduction_ci95": bootstrap_ci(
                first_collision - selected_collision, args.seed + 1, args.bootstrap),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
