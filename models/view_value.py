"""Predict expected task-risk reduction for discrete camera candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ViewValueOutput:
    mean_risk_reduction: Tensor  # [B,V]
    log_variance: Tensor         # [B,V]

    @property
    def std(self) -> Tensor:
        return (0.5 * self.log_variance).exp()


class ViewValueModel(nn.Module):
    def __init__(self, belief_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.view_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.risk_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(belief_dim + 2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, belief: Tensor, candidate_views: Tensor, action_risk: Tensor) -> ViewValueOutput:
        if candidate_views.ndim != 3 or candidate_views.shape[-1] != 6:
            raise ValueError("candidate_views must have shape [B,V,6]")
        if action_risk.ndim != 2:
            raise ValueError("action_risk must have shape [B,M]")
        stats = torch.stack((
            action_risk.min(dim=-1).values,
            action_risk.mean(dim=-1),
            action_risk.std(dim=-1, correction=0),
            action_risk.max(dim=-1).values,
        ), dim=-1)
        v = candidate_views.shape[1]
        context = torch.cat((
            belief[:, None].expand(-1, v, -1),
            self.view_encoder(candidate_views),
            self.risk_encoder(stats)[:, None].expand(-1, v, -1),
        ), dim=-1)
        value = self.head(context)
        return ViewValueOutput(value[..., 0], value[..., 1].clamp(-8.0, 8.0))


def view_value_loss(output: ViewValueOutput, target: Tensor) -> Tensor:
    error = (output.mean_risk_reduction - target).square()
    return (0.5 * (error * torch.exp(-output.log_variance) + output.log_variance)).mean()
