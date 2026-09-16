"""Fit temperature and one-sided conformal residuals on calibration episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import RiskCalibration, load_risk4d_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage", type=float, default=0.9)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_risk4d_checkpoint(path, device) for path in args.checkpoints]
    dataset = Risk4DDataset(args.data)
    success_logits, collision_logits, success, collision = [], [], [], []
    with torch.inference_mode():
        for item in dataset:
            batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
            outputs = [model(
                batch["points"], batch["point_features"], batch["robot_state"],
                batch["robot_action_flow"], batch["task_embedding"],
            ) for model in models]
            success_logits.append(torch.stack([output.success_logits for output in outputs]).mean(0).flatten())
            collision_logits.append(torch.stack([output.collision_logits for output in outputs]).mean(0).flatten())
            success.append(batch["success"].flatten())
            collision.append(batch["collision"].flatten())
    calibration = RiskCalibration.fit(
        torch.cat(success_logits), torch.cat(collision_logits),
        torch.cat(success), torch.cat(collision), args.coverage,
    )
    calibration.save(args.output)
    print(Path(args.output).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
