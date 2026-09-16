"""Label active views by realized utility change after risk-based reranking."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import RiskCalibration, load_risk4d_checkpoint


def predicted_risk(models, calibration, points, features, robot_state,
                   robot_action_flow, task_embedding, failure_weight, collision_weight):
    outputs = [model(
        points, features, robot_state, robot_action_flow, task_embedding,
    ) for model in models]
    success_logits = torch.stack([output.success_logits for output in outputs]).mean(0)
    collision_logits = torch.stack([output.collision_logits for output in outputs]).mean(0)
    probabilities = torch.stack([
        torch.stack((output.success_logits.sigmoid(), output.collision_logits.sigmoid()), -1)
        for output in outputs
    ])
    epistemic = probabilities.var(0, correction=0).mean(-1)
    return calibration.risk_bound(
        success_logits, collision_logits, epistemic,
        failure_weight, collision_weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--failure-weight", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=1.0)
    args = parser.parse_args()
    source, destination = Path(args.input), Path(args.output)
    files = sorted(source.glob("**/*.npz"))
    if not files:
        raise ValueError(f"no NPZ clips found in {source}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_risk4d_checkpoint(path, device) for path in args.checkpoints]
    calibration = RiskCalibration.load(args.calibration)
    with torch.inference_mode():
        for file_path in files:
            with np.load(file_path, allow_pickle=False) as archive:
                sample = {key: np.array(archive[key], copy=True) for key in archive.files}
            for key in ("view_points", "view_point_features", "robot_action_flow"):
                if key not in sample:
                    raise ValueError(f"{file_path} is missing {key}")
            tensor = lambda key: torch.from_numpy(sample[key]).float().unsqueeze(0).to(device)
            shared = (
                tensor("robot_state"), tensor("robot_action_flow"), tensor("task_embedding"),
            )
            base_risk = predicted_risk(
                models, calibration, tensor("points"), tensor("point_features"), *shared,
                args.failure_weight, args.collision_weight,
            )[0]
            actual_utility = (
                args.failure_weight * (1.0 - torch.from_numpy(sample["success"]).float())
                + args.collision_weight * torch.from_numpy(sample["collision"]).float()
            )
            base_utility = actual_utility[base_risk.argmin().cpu()]
            reductions = []
            for view_index in range(len(sample["view_points"])):
                view_points = torch.from_numpy(sample["view_points"][view_index]).float()[None].to(device)
                view_features = torch.from_numpy(sample["view_point_features"][view_index]).float()[None].to(device)
                view_risk = predicted_risk(
                    models, calibration, view_points, view_features, *shared,
                    args.failure_weight, args.collision_weight,
                )[0]
                view_utility = actual_utility[view_risk.argmin().cpu()]
                reductions.append(float(base_utility - view_utility))
            sample["view_risk_reduction"] = np.asarray(reductions, np.float32)
            output_file = destination / file_path.relative_to(source)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, **sample)
            print(f"wrote {output_file} view_risk_reduction={reductions}")


if __name__ == "__main__":
    main()
