"""PointWorld-like action-conditioned 4D perception adapter.

The visual geometry backbone is external and frozen.  This module only learns
to encode the cached point-token history and robot surface flow, then predicts
future scene point flow and per-point log variance.  It intentionally has no
task-success or collision heads so its geometry quality can be measured alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class PointWorld4DOutput:
    future_point_flow: Tensor       # [B,M,H,N,3]
    flow_log_variance: Tensor       # [B,M,H,N]


class PointWorld4DModel(nn.Module):
    """Compact action-conditioned point-flow predictor."""

    def __init__(
        self,
        point_feature_dim: int = 1,
        robot_state_dim: int = 16,
        hidden_dim: int = 256,
        future_steps: int = 8,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(3 + point_feature_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        # Mean/max xyz statistics retain the robot's spatial trajectory while
        # keeping the token budget independent of the sampled surface size Q.
        self.robot_flow_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.scene_temporal = nn.TransformerEncoder(temporal_layer, num_layers=layers)
        self.action_temporal = nn.TransformerEncoder(
            temporal_layer, num_layers=max(1, layers // 2))
        self.candidate_temporal = nn.TransformerEncoder(
            temporal_layer, num_layers=max(1, layers // 2))
        self.point_decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, future_steps * 4),
        )

    def forward(
        self,
        points: Tensor,
        point_features: Tensor,
        robot_state: Tensor,
        robot_action_flow: Tensor,
    ) -> PointWorld4DOutput:
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("points must have shape [B,K,N,3]")
        if point_features.shape[:3] != points.shape[:3]:
            raise ValueError("point_features must align with points in B,K,N")
        if robot_action_flow.ndim != 5 or robot_action_flow.shape[-1] != 3:
            raise ValueError("robot_action_flow must have shape [B,M,L,Q,3]")
        if robot_state.ndim == 2:
            robot_state = robot_state[:, None]
        if robot_state.ndim != 3 or robot_state.shape[0] != points.shape[0]:
            raise ValueError("robot_state must have shape [B,K,S] or [B,S]")

        b, _, n, _ = points.shape
        m, action_steps = robot_action_flow.shape[1:3]
        encoded_points = self.point_encoder(torch.cat((points, point_features), dim=-1))
        confidence = point_features[..., :1].detach().clamp(0.05, 1.0)
        frame_tokens = (encoded_points * confidence).sum(2) / confidence.sum(2).clamp_min(1e-4)
        scene = self.scene_temporal(frame_tokens)[:, -1]
        scene = scene + self.robot_encoder(robot_state[:, -1])

        flow_min = robot_action_flow.amin(dim=3)
        flow_max = robot_action_flow.amax(dim=3)
        flow_stats = torch.cat((flow_min, flow_max), dim=-1)
        flow_tokens = self.robot_flow_encoder(flow_stats).reshape(
            b * m, action_steps, self.hidden_dim)
        action = self.action_temporal(flow_tokens).mean(1).reshape(b, m, self.hidden_dim)
        context = self.candidate_temporal(scene[:, None] + action)

        current_points = encoded_points[:, -1]
        point_context = torch.cat((
            context[:, :, None].expand(-1, -1, n, -1),
            current_points[:, None].expand(-1, m, -1, -1),
        ), dim=-1)
        decoded = self.point_decoder(point_context).view(
            b, m, n, self.future_steps, 4).permute(0, 1, 3, 2, 4)
        return PointWorld4DOutput(
            future_point_flow=decoded[..., :3],
            flow_log_variance=decoded[..., 3].clamp(-8.0, 8.0),
        )


def pointworld4d_loss(output: PointWorld4DOutput, target: Tensor) -> dict[str, Tensor]:
    """Heteroscedastic flow loss with a separately logged Smooth-L1 term."""
    error = F.smooth_l1_loss(output.future_point_flow, target, reduction="none").mean(-1)
    nll = 0.5 * (error * torch.exp(-output.flow_log_variance)
                 + output.flow_log_variance).mean()
    smooth_l1 = error.mean()
    return {"total": nll, "flow_nll": nll, "flow_smooth_l1": smooth_l1}
