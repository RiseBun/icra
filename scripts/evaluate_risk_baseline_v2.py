"""Evaluate scalar v2 controls with the same candidate labels and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import ScalarCandidateRiskModel
from scripts.evaluate_risk4d import auroc, ece


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ScalarCandidateRiskModel(**state["model_config"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    dataset = Risk4DDataset(args.data)
    success_probability, collision_probability, successes, collisions = [], [], [], []
    selected, first = [], []
    with torch.inference_mode():
        for item in dataset:
            batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
            output = model(
                batch["points"], batch["point_features"], batch["robot_state"],
                batch["robot_action_flow"], batch["task_embedding"],
            )
            sp = output.success_logits[0].sigmoid().cpu().numpy()
            cp = output.collision_logits[0].sigmoid().cpu().numpy()
            actual = 1.0 - item["success"].numpy() + item["collision"].numpy()
            predicted = 1.0 - sp + cp
            success_probability.append(sp); collision_probability.append(cp)
            successes.append(item["success"].numpy()); collisions.append(item["collision"].numpy())
            selected.append(actual[predicted.argmin()]); first.append(actual[0])
    sp, cp = np.concatenate(success_probability), np.concatenate(collision_probability)
    success, collision = np.concatenate(successes), np.concatenate(collisions)
    result = {
        "baseline_type": state["baseline_type"], "clips": len(dataset),
        "failure_auroc": auroc(1.0 - sp, 1.0 - success),
        "collision_auroc": auroc(cp, collision),
        "success_brier": float(np.mean((sp - success) ** 2)),
        "collision_brier": float(np.mean((cp - collision) ** 2)),
        "success_ece": ece(sp, success), "collision_ece": ece(cp, collision),
        "selected_utility": float(np.mean(selected)),
        "first_sample_utility": float(np.mean(first)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
