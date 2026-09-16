"""Risk head that consumes a frozen PointWorld-like future 4D belief."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RiskFrom4DOutput:
    task_contact_logits: Tensor       # [B,M,H,N]
    harmful_collision_logits: Tensor  # [B,M,H,N]
    success_logits: Tensor             # [B,M]
    collision_logits: Tensor           # [B,M]


class RiskFrom4DBeliefModel(nn.Module):
    """Predict spatial and candidate risk from precomputed future point flow."""

    def __init__(
        self,
        point_feature_dim: int = 1,
        robot_state_dim: int = 16,
        hidden_dim: int = 256,
        future_steps: int = 8,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(3 + point_feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.robot_flow_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.future_flow_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.scene_temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.action_temporal = nn.TransformerEncoder(
            layer, num_layers=max(1, layers // 2))
        self.future_temporal = nn.TransformerEncoder(
            layer, num_layers=max(1, layers // 2))
        self.candidate_temporal = nn.TransformerEncoder(
            layer, num_layers=max(1, layers // 2))
        self.spatial_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2))
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2))

    def forward(
        self,
        points: Tensor,
        point_features: Tensor,
        robot_state: Tensor,
        robot_action_flow: Tensor,
        future_point_flow: Tensor,
    ) -> RiskFrom4DOutput:
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("points must have shape [B,K,N,3]")
        if point_features.shape[:3] != points.shape[:3]:
            raise ValueError("point_features must align with points in B,K,N")
        if robot_action_flow.ndim != 5 or robot_action_flow.shape[-1] != 3:
            raise ValueError("robot_action_flow must have shape [B,M,L,Q,3]")
        if future_point_flow.ndim != 5 or future_point_flow.shape[-1] != 3:
            raise ValueError("future_point_flow must have shape [B,M,H,N,3]")
        if future_point_flow.shape[0] != points.shape[0] or future_point_flow.shape[3] != points.shape[2]:
            raise ValueError("future point flow must align with batch and point count")
        if robot_state.ndim == 2:
            robot_state = robot_state[:, None]
        if robot_state.ndim != 3 or robot_state.shape[0] != points.shape[0]:
            raise ValueError("robot_state must have shape [B,K,S] or [B,S]")

        b, _, n, _ = points.shape
        m, h = future_point_flow.shape[1:3]
        point_tokens = self.point_encoder(torch.cat((points, point_features), dim=-1))
        confidence = point_features[..., :1].detach().clamp(0.05, 1.0)
        frame_tokens = (point_tokens * confidence).sum(2) / confidence.sum(2).clamp_min(1e-4)
        scene = self.scene_temporal(frame_tokens)[:, -1] + self.state_encoder(robot_state[:, -1])

        robot_stats = torch.cat((robot_action_flow.amin(3), robot_action_flow.amax(3)), dim=-1)
        action_tokens = self.robot_flow_encoder(robot_stats).reshape(
            b * m, robot_stats.shape[2], self.hidden_dim)
        action = self.action_temporal(action_tokens).mean(1).reshape(b, m, self.hidden_dim)

        flow_tokens = self.future_flow_encoder(future_point_flow)
        future = flow_tokens.mean(3).reshape(b * m, h, self.hidden_dim)
        future = self.future_temporal(future).mean(1).reshape(b, m, self.hidden_dim)
        context = self.candidate_temporal(scene[:, None] + action + future)

        current = point_tokens[:, -1]
        flow_point = self.future_flow_encoder(future_point_flow)
        spatial_context = torch.cat((
            context[:, :, None, None].expand(-1, -1, h, n, -1),
            current[:, None, None].expand(-1, m, h, -1, -1),
            flow_point,
        ), dim=-1)
        spatial = self.spatial_head(spatial_context)
        contact, harmful = spatial[..., 0], spatial[..., 1]
        contact_prob, harmful_prob = contact.sigmoid(), harmful.sigmoid()
        evidence = torch.stack((
            contact_prob.mean((-1, -2)), contact_prob.amax((-1, -2)),
            harmful_prob.mean((-1, -2)), harmful_prob.amax((-1, -2)),
            future_point_flow.norm(dim=-1).mean((-1, -2)),
            future_point_flow.var(dim=-1, unbiased=False).mean((-1, -2)),
        ), dim=-1)
        risk = self.risk_head(torch.cat((context, evidence), dim=-1))
        return RiskFrom4DOutput(contact, harmful, risk[..., 0], risk[..., 1])


def risk_from_4d_loss(output: RiskFrom4DOutput, contact_target: Tensor,
                      collision_map_target: Tensor, success_target: Tensor,
                      collision_target: Tensor) -> dict[str, Tensor]:
    """Risk-head objective; the future-flow predictor is intentionally frozen."""
    zero = output.success_logits.sum() * 0.0
    losses = {
        "contact": F.binary_cross_entropy_with_logits(output.task_contact_logits, contact_target),
        "spatial_collision": F.binary_cross_entropy_with_logits(
            output.harmful_collision_logits, collision_map_target),
        "success": F.binary_cross_entropy_with_logits(output.success_logits, success_target),
        "collision": F.binary_cross_entropy_with_logits(output.collision_logits, collision_target),
        "success_rank": zero,
        "collision_rank": zero,
    }
    for name, logits, labels in (
        ("success_rank", output.success_logits, success_target),
        ("collision_rank", output.collision_logits, collision_target),
    ):
        pair = labels[:, :, None] - labels[:, None, :]
        mask = pair > 0
        if mask.any():
            losses[name] = F.relu(0.2 - (logits[:, :, None] - logits[:, None, :])[mask]).mean()
    losses["total"] = (losses["contact"] + 2.0 * losses["spatial_collision"]
                       + 2.0 * losses["success"] + 2.0 * losses["collision"]
                       + 0.5 * (losses["success_rank"] + losses["collision_rank"]))
    return losses
