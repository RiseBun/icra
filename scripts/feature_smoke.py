"""Run the trainable head on one exported backbone feature file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_file")
    args = parser.parse_args()
    data = np.load(args.feature_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.from_numpy(data["points"]).float().unsqueeze(0).to(device)
    actions = torch.randn(1, 8, 10, device=device)
    robot_state = torch.zeros(1, 16, device=device)
    model = ActionConditioned4DModel(
        robot_state_dim=16, action_dim=10, future_steps=8,
        hidden_dim=256, layers=4, heads=8,
    ).to(device).eval()
    with torch.inference_mode():
        output = model(points, actions, robot_state)
    print({
        "device": str(device),
        "backbone": str(data["backbone"]),
        "points": tuple(points.shape),
        "future_delta": tuple(output.future_delta.shape),
        "success": tuple(output.success_logits.shape),
    })


if __name__ == "__main__":
    main()
