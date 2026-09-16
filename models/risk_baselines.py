"""Strict v2 controls for action-only and geometry-scalar candidate risk."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ScalarRiskOutput:
    belief: Tensor
    success_logits: Tensor
    collision_logits: Tensor


class ScalarCandidateRiskModel(nn.Module):
    """Candidate risk without spatial future/contact/collision-map prediction."""

    def __init__(
        self,
        point_feature_dim: int,
        robot_state_dim: int,
        task_embedding_dim: int,
        hidden_dim: int = 512,
        layers: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        include_geometry: bool = True,
    ) -> None:
        super().__init__()
        self.include_geometry = include_geometry
        self.hidden_dim = hidden_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(3 + point_feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        ) if include_geometry else None
        self.state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_embedding_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, 4 * hidden_dim, dropout, "gelu",
            batch_first=True, norm_first=True,
        )
        self.scene_temporal = nn.TransformerEncoder(layer, layers) if include_geometry else None
        self.action_temporal = nn.TransformerEncoder(layer, max(1, layers // 2))
        self.candidate_encoder = nn.TransformerEncoder(layer, max(1, layers // 2))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )

    def forward(self, points, point_features, robot_state, robot_action_flow, task_embedding):
        if robot_state.ndim == 2:
            robot_state = robot_state[:, None]
        context = self.state_encoder(robot_state[:, -1]) + self.task_encoder(task_embedding)
        if self.include_geometry:
            point_token = self.point_encoder(torch.cat((points, point_features), -1))
            confidence = point_features[..., :1].detach().clamp(0.05, 1.0)
            frame = (point_token * confidence).sum(2) / confidence.sum(2).clamp_min(1e-4)
            context = context + self.scene_temporal(frame)[:, -1]
        b, m, steps = robot_action_flow.shape[:3]
        action = self.action_encoder(robot_action_flow).mean(3).reshape(b * m, steps, self.hidden_dim)
        action = self.action_temporal(action).mean(1).reshape(b, m, self.hidden_dim)
        candidate = self.candidate_encoder(context[:, None] + action)
        logits = self.head(candidate)
        return ScalarRiskOutput(context, logits[..., 0], logits[..., 1])


def scalar_risk_loss(output: ScalarRiskOutput, success: Tensor, collision: Tensor) -> Tensor:
    success_loss = F.binary_cross_entropy_with_logits(output.success_logits, success)
    collision_loss = F.binary_cross_entropy_with_logits(output.collision_logits, collision)
    return 2.0 * success_loss + 2.0 * collision_loss
