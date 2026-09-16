"""One-step offline active-view comparison on held-out v2 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import RiskCalibration, ViewValueModel, load_risk4d_checkpoint


def infer(models, calibration, points, features, robot_state, robot_flow, task_embedding):
    outputs = [model(points, features, robot_state, robot_flow, task_embedding) for model in models]
    success_logits = torch.stack([output.success_logits for output in outputs]).mean(0)
    collision_logits = torch.stack([output.collision_logits for output in outputs]).mean(0)
    member_probability = torch.stack([
        torch.stack((output.success_logits.sigmoid(), output.collision_logits.sigmoid()), -1)
        for output in outputs
    ])
    epistemic = member_probability.var(0, correction=0).mean(-1)
    risk = calibration.risk_bound(success_logits, collision_logits, epistemic)
    belief = torch.stack([output.belief for output in outputs]).mean(0)
    return risk, belief, epistemic


def load_view_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ViewValueModel(**state["model_config"])
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def summarize(records):
    result = {}
    for name in records[0]:
        rows = [record[name] for record in records]
        executed = np.asarray([row["executed"] for row in rows], bool)
        result[name] = {
            "coverage": float(executed.mean()),
            "observations": float(np.mean([row["observations"] for row in rows])),
            "utility_on_executed": float(np.mean([
                row["utility"] for row in rows if row["executed"]
            ])) if executed.any() else None,
            "success_on_executed": float(np.mean([
                row["success"] for row in rows if row["executed"]
            ])) if executed.any() else None,
            "collision_on_executed": float(np.mean([
                row["collision"] for row in rows if row["executed"]
            ])) if executed.any() else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--risk-checkpoints", nargs="+", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--view-checkpoint", required=True)
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    parser.add_argument("--confidence-z", type=float, default=1.645)
    parser.add_argument("--camera-cost-weight", type=float, default=0.05)
    parser.add_argument("--camera-safety-weight", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_risk4d_checkpoint(path, device) for path in args.risk_checkpoints]
    calibration = RiskCalibration.load(args.calibration)
    view_model = load_view_model(args.view_checkpoint, device)
    dataset = Risk4DDataset(args.data)
    records = []
    with torch.inference_mode():
        for item in dataset:
            for key in ("view_points", "view_point_features", "candidate_views",
                        "view_cost", "view_safety_cost"):
                if key not in item:
                    raise ValueError(f"active-view evaluation requires {key}")
            batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
            base_risk, belief, _ = infer(
                models, calibration, batch["points"], batch["point_features"],
                batch["robot_state"], batch["robot_action_flow"], batch["task_embedding"],
            )
            view_risks, view_uncertainty = [], []
            for view_index in range(item["view_points"].shape[0]):
                risk, _, uncertainty = infer(
                    models, calibration,
                    batch["view_points"][:, view_index],
                    batch["view_point_features"][:, view_index],
                    batch["robot_state"], batch["robot_action_flow"], batch["task_embedding"],
                )
                view_risks.append(risk[0])
                view_uncertainty.append(uncertainty[0])
            view_risks = torch.stack(view_risks)
            view_uncertainty = torch.stack(view_uncertainty)
            success, collision = item["success"].numpy(), item["collision"].numpy()
            utility = 1.0 - success + collision
            view_actions = view_risks.argmin(-1).cpu().numpy()
            view_utility = utility[view_actions]
            base_action = int(base_risk[0].argmin())

            confidence = item["view_point_features"][..., 0].mean(dim=(-1, -2)).numpy()
            base_points = item["points"][-1]
            novelty = []
            for view in item["view_points"][:, -1]:
                novelty.append(float(torch.cdist(view, base_points).min(dim=-1).values.mean()))
            uncertainty_score = -view_uncertainty.gather(
                1, torch.from_numpy(view_actions).to(device)[:, None]).squeeze(1).cpu().numpy()
            view_output = view_model(belief, batch["candidate_views"], base_risk)
            view_value = view_output.mean_risk_reduction[0]
            learned_score = (
                view_value
                - args.confidence_z * view_output.std[0]
                - args.camera_cost_weight * batch["view_cost"][0]
                - args.camera_safety_weight * batch["view_safety_cost"][0]
            ).cpu().numpy()
            selections = {
                "random_view": None,
                "maximum_visible_area": int(np.argmax(confidence)),
                "geometric_information_gain": int(np.argmax(novelty)),
                "uncertainty_reduction": int(np.argmax(uncertainty_score)),
                "learned_risk_reduction": int(np.argmax(learned_score)),
                "oracle_view": int(np.argmin(view_utility)),
            }
            row = {
                "fixed_view": {
                    "utility": float(utility[base_action]), "success": float(success[base_action]),
                    "collision": float(collision[base_action]), "observations": 0, "executed": True,
                }
            }
            for name, view_index in selections.items():
                if view_index is None:
                    row[name] = {
                        "utility": float(view_utility.mean()),
                        "success": float(success[view_actions].mean()),
                        "collision": float(collision[view_actions].mean()),
                        "observations": 1, "executed": True,
                    }
                else:
                    action = int(view_actions[view_index])
                    row[name] = {
                        "utility": float(utility[action]), "success": float(success[action]),
                        "collision": float(collision[action]), "observations": 1, "executed": True,
                    }
            if float(base_risk[0, base_action]) <= args.risk_threshold:
                action, observations, executed = base_action, 0, True
            else:
                selected_view = selections["learned_risk_reduction"]
                action, observations = int(view_actions[selected_view]), 1
                executed = float(view_risks[selected_view, action]) <= args.risk_threshold
            row["ours_selective"] = {
                "utility": float(utility[action]), "success": float(success[action]),
                "collision": float(collision[action]), "observations": observations,
                "executed": bool(executed),
            }
            for external in ("tavp_view_score", "ota_view_score"):
                if external in item:
                    view_index = int(item[external].argmax())
                    action = int(view_actions[view_index])
                    row[external.removesuffix("_view_score")] = {
                        "utility": float(utility[action]), "success": float(success[action]),
                        "collision": float(collision[action]), "observations": 1, "executed": True,
                    }
            records.append(row)
    print(json.dumps({"clips": len(records), "strategies": summarize(records)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
