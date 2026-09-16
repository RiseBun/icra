"""Evaluate risk from a frozen PointWorld-like future-flow predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import PointWorld4DModel, RiskFrom4DBeliefModel
from scripts.evaluate_risk4d import auroc, average_precision, binary_iou, ece


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--perception-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pstate = torch.load(args.perception_checkpoint, map_location="cpu", weights_only=True)
    perception = PointWorld4DModel(**pstate["model_config"]).to(device)
    perception.load_state_dict(pstate["model"]); perception.eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = RiskFrom4DBeliefModel(**state["model_config"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    dataset = Risk4DDataset(args.data)
    pred_success, pred_collision, success, collision = [], [], [], []
    contact, contact_target, harmful, harmful_target = [], [], [], []
    flow_epe, dynamic_epe = [], []
    with torch.inference_mode():
        for item in dataset:
            batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
            predicted_flow = perception(
                batch["points"], batch["point_features"], batch["robot_state"],
                batch["robot_action_flow"],
            ).future_point_flow
            output = model(
                batch["points"], batch["point_features"], batch["robot_state"],
                batch["robot_action_flow"], predicted_flow,
            )
            pred_success.append(output.success_logits[0].sigmoid().cpu().numpy())
            pred_collision.append(output.collision_logits[0].sigmoid().cpu().numpy())
            success.append(item["success"].numpy()); collision.append(item["collision"].numpy())
            contact.append(output.task_contact_logits[0].sigmoid().cpu().numpy())
            contact_target.append(item["task_contact_map"].numpy())
            harmful.append(output.harmful_collision_logits[0].sigmoid().cpu().numpy())
            harmful_target.append(item["harmful_collision_map"].numpy())
            error = np.linalg.norm(predicted_flow[0].cpu().numpy() - item["future_point_flow"].numpy(), axis=-1)
            flow_epe.append(error.reshape(-1))
            dynamic = np.linalg.norm(item["future_point_flow"].numpy(), axis=-1) > 0.01
            if dynamic.any(): dynamic_epe.append(error[dynamic])
    ps, pc = np.concatenate(pred_success), np.concatenate(pred_collision)
    sy, cy = np.concatenate(success), np.concatenate(collision)
    ct, cty = np.concatenate(contact).reshape(-1), np.concatenate(contact_target).reshape(-1)
    hp, hty = np.concatenate(harmful).reshape(-1), np.concatenate(harmful_target).reshape(-1)
    result = {
        "clips": len(dataset),
        "geometry": {
            "flow_epe": float(np.concatenate(flow_epe).mean()),
            "dynamic_flow_epe": float(np.concatenate(dynamic_epe).mean()) if dynamic_epe else float("nan"),
        },
        "spatial_risk": {
            "contact_auprc": average_precision(ct, cty),
            "contact_miou": binary_iou(ct, cty),
            "harmful_collision_auprc": average_precision(hp, hty),
            "harmful_collision_miou": binary_iou(hp, hty),
        },
        "candidate_risk": {
            "failure_auroc": auroc(1.0 - ps, 1.0 - sy),
            "collision_auroc": auroc(pc, cy),
            "success_brier": float(np.mean((ps - sy) ** 2)),
            "collision_brier": float(np.mean((pc - cy) ** 2)),
            "success_ece": ece(ps, sy), "collision_ece": ece(pc, cy),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"; print(text, end="")
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)


if __name__ == "__main__":
    main()
