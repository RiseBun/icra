"""Deterministic future-motion baseline."""

from __future__ import annotations

import torch
from torch import Tensor


def predict_constant_velocity(points: Tensor, future_steps: int) -> Tensor:
    """Predict [B,H,N,3] future points from [B,K,N,3] history."""
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,K,N,3]")
    velocity = points[:, -1] - points[:, -2]
    steps = torch.arange(
        1, future_steps + 1, device=points.device, dtype=points.dtype
    ).view(1, future_steps, 1, 1)
    return points[:, -1:, :, :] + steps * velocity[:, None, :, :]
