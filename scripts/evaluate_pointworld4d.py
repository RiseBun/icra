"""Evaluate geometry-only 4D flow EPE on a held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import PointWorld4DModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PointWorld4DModel(**state["model_config"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    dataset = Risk4DDataset(args.data)
    all_epe, dynamic_epe, uncertainty_error = [], [], []
    with torch.inference_mode():
        for item in dataset:
            batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
            output = model(batch["points"], batch["point_features"],
                           batch["robot_state"], batch["robot_action_flow"])
            prediction = output.future_point_flow[0].cpu().numpy()
            target = item["future_point_flow"].numpy()
            epe = np.linalg.norm(prediction - target, axis=-1)
            dynamic = np.linalg.norm(target, axis=-1) > 0.01
            all_epe.append(epe.reshape(-1))
            if dynamic.any():
                dynamic_epe.append(epe[dynamic])
            uncertainty_error.append(np.corrcoef(
                output.flow_log_variance[0].exp().cpu().numpy().reshape(-1),
                epe.reshape(-1))[0, 1])
    result = {
        "clips": len(dataset),
        "flow_epe": float(np.concatenate(all_epe).mean()),
        "dynamic_flow_epe": float(np.concatenate(dynamic_epe).mean()) if dynamic_epe else float("nan"),
        "uncertainty_error_correlation": float(np.nanmean(uncertainty_error)),
        "checkpoint": str(args.checkpoint),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)


if __name__ == "__main__":
    main()
