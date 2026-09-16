"""Evaluate geometry and action-use tests for Omega-ActionWorld.

The evaluator compares normal conditioning with shuffled candidates and a zero
action control.  It also measures whether pairwise predicted future changes
track the pairwise changes in the recorded simulator latent targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.omega_actionworld import OmegaActionWorld


def _corr(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or np.std(values_a) < 1e-8 or np.std(values_b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(values_a, values_b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    files = sorted(Path(args.data).glob("**/*.npz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError(f"no cached latent files in {args.data}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = OmegaActionWorld(**state["model_config"]).eval()
    normal_epe, shuffled_epe, zero_epe = [], [], []
    normal_residual_epe, shuffled_residual_epe, zero_residual_epe = [], [], []
    predicted_delta, target_delta = [], []
    generator = torch.Generator().manual_seed(2027)
    with torch.inference_mode():
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                required = ("omega_latent_tokens", "omega_future_latent_tokens",
                            "robot_state", "robot_action_flow")
                missing = [key for key in required if key not in archive]
                if missing:
                    raise ValueError(f"{path} is missing {missing}")
                tokens = torch.from_numpy(archive["omega_latent_tokens"])[None].float()
                target = torch.from_numpy(archive["omega_future_latent_tokens"])[None].float()
                robot_state = torch.from_numpy(archive["robot_state"])[None].float()
                robot_flow = torch.from_numpy(archive["robot_action_flow"])[None].float()
            normal = model(tokens, robot_state, robot_flow).future_tokens
            zero = model(tokens, robot_state, torch.zeros_like(robot_flow)).future_tokens
            permutation = torch.randperm(robot_flow.shape[1], generator=generator)
            shuffled = model(tokens, robot_state, robot_flow[:, permutation]).future_tokens
            normal_epe.append(float((normal - target).abs().mean()))
            zero_epe.append(float((zero - target).abs().mean()))
            shuffled_epe.append(float((shuffled - target).abs().mean()))
            target_centered = target - target.mean(dim=1, keepdim=True)
            for prediction, bucket in ((normal, normal_residual_epe),
                                       (zero, zero_residual_epe),
                                       (shuffled, shuffled_residual_epe)):
                centered = prediction - prediction.mean(dim=1, keepdim=True)
                bucket.append(float((centered - target_centered).abs().mean()))
            # Compare candidate-pair distances in the future latent trajectory.
            pred = normal[0].flatten(1).norm(dim=-1)
            true = target[0].flatten(1).norm(dim=-1)
            for i in range(pred.shape[0]):
                for j in range(i + 1, pred.shape[0]):
                    predicted_delta.append(float((normal[0, i] - normal[0, j]).abs().mean()))
                    target_delta.append(float((target[0, i] - target[0, j]).abs().mean()))
    result = {
        "clips": len(files),
        "latent_mae_normal": float(np.mean(normal_epe)),
        "latent_mae_zero_action": float(np.mean(zero_epe)),
        "latent_mae_shuffled_action": float(np.mean(shuffled_epe)),
        "shuffle_degradation": float(np.mean(shuffled_epe) - np.mean(normal_epe)),
        "action_residual_mae_normal": float(np.mean(normal_residual_epe)),
        "action_residual_mae_zero_action": float(np.mean(zero_residual_epe)),
        "action_residual_mae_shuffled_action": float(np.mean(shuffled_residual_epe)),
        "action_residual_shuffle_degradation": float(
            np.mean(shuffled_residual_epe) - np.mean(normal_residual_epe)),
        "action_effect_correlation": _corr(predicted_delta, target_delta),
        "checkpoint": str(args.checkpoint),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)


if __name__ == "__main__":
    main()
