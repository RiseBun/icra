"""Checkpoint I/O shared by v2 training, calibration and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from .risk4d import RiskConditioned4DModel


def load_risk4d_checkpoint(path: str | Path, device: torch.device) -> RiskConditioned4DModel:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = RiskConditioned4DModel(**state["model_config"])
    model.load_state_dict(state["model"])
    return model.to(device).eval()
