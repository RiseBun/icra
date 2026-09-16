"""Action-conditioned 4D risk maps from frozen geometry features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Risk4DOutput:
    belief: Tensor                     # [B,D]
    future_point_flow: Tensor          # [B,M,H,N,3]
    flow_log_variance: Tensor          # [B,M,H,N]
    task_contact_logits: Tensor        # [B,M,H,N]
    harmful_collision_logits: Tensor   # [B,M,H,N]
    success_logits: Tensor             # [B,M]
    collision_logits: Tensor           # [B,M]

    @property
    def aleatoric_uncertainty(self) -> Tensor:
        return self.flow_log_variance.exp().mean(dim=(-1, -2))


class RiskConditioned4DModel(nn.Module):
    """Predict geometry and task risk for 3D robot-action-flow candidates."""

    def __init__(
        self,
        point_feature_dim: int = 1,
        robot_state_dim: int = 16,
        task_embedding_dim: int = 32,
        hidden_dim: int = 512,
        future_steps: int = 8,
        layers: int = 6,
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
        self.robot_state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_embedding_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.robot_point_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.scene_temporal = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.action_temporal = nn.TransformerEncoder(encoder_layer, num_layers=max(1, layers // 2))
        self.candidate_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, layers // 2))
        self.point_decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, future_steps * 6),
        )
        # Context plus spatial evidence: mean/max contact, collision, motion and variance.
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        points: Tensor,
        point_features: Tensor,
        robot_state: Tensor,
        robot_action_flow: Tensor,
        task_embedding: Tensor,
    ) -> Risk4DOutput:
        """Run the model using the v2 feature contract.

        Shapes are points/features [B,K,N,*], robot_state [B,K,S] or [B,S],
        robot_action_flow [B,M,L,Q,3], and task_embedding [B,D].
        """
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
        point_tokens = self.point_encoder(torch.cat((points, point_features), dim=-1))
        confidence = point_features[..., :1].detach().clamp(0.05, 1.0)
        frame_tokens = (point_tokens * confidence).sum(2) / confidence.sum(2).clamp_min(1e-4)
        scene = self.scene_temporal(frame_tokens)[:, -1]
        scene = scene + self.robot_state_encoder(robot_state[:, -1])
        scene = scene + self.task_encoder(task_embedding)

        action_points = self.robot_point_encoder(robot_action_flow)
        action_steps_tokens = action_points.mean(dim=3).reshape(b * m, action_steps, self.hidden_dim)
        action = self.action_temporal(action_steps_tokens).mean(dim=1).reshape(b, m, self.hidden_dim)
        context = self.candidate_encoder(scene[:, None] + action)

        current_points = point_tokens[:, -1]
        point_context = torch.cat((
            context[:, :, None].expand(-1, -1, n, -1),
            current_points[:, None].expand(-1, m, -1, -1),
        ), dim=-1)
        decoded = self.point_decoder(point_context).view(b, m, n, self.future_steps, 6)
        decoded = decoded.permute(0, 1, 3, 2, 4)
        flow = decoded[..., :3]
        log_var = decoded[..., 3].clamp(-8.0, 8.0)
        contact = decoded[..., 4]
        spatial_collision = decoded[..., 5]

        contact_prob = contact.sigmoid()
        collision_prob = spatial_collision.sigmoid()
        motion = flow.norm(dim=-1)
        evidence = torch.stack((
            contact_prob.mean(dim=(-1, -2)), contact_prob.amax(dim=(-1, -2)),
            collision_prob.mean(dim=(-1, -2)), collision_prob.amax(dim=(-1, -2)),
            motion.mean(dim=(-1, -2)), log_var.exp().mean(dim=(-1, -2)),
        ), dim=-1)
        risk = self.risk_head(torch.cat((context, evidence), dim=-1))
        return Risk4DOutput(
            belief=scene, future_point_flow=flow, flow_log_variance=log_var,
            task_contact_logits=contact,
            harmful_collision_logits=spatial_collision,
            success_logits=risk[..., 0], collision_logits=risk[..., 1],
        )


def _pairwise_rank_loss(logits: Tensor, labels: Tensor, higher_is_positive: bool) -> Tensor:
    difference = labels[:, :, None] - labels[:, None, :]
    mask = difference > 0
    if not mask.any():
        return logits.sum() * 0.0
    signed = logits[:, :, None] - logits[:, None, :]
    if not higher_is_positive:
        signed = -signed
    return F.relu(0.2 - signed[mask]).mean()


def risk4d_loss(
    output: Risk4DOutput,
    future_point_flow: Tensor,
    task_contact_map: Tensor,
    harmful_collision_map: Tensor,
    success: Tensor,
    collision: Tensor,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Tensor]:
    """Joint heteroscedastic flow, spatial-risk and candidate-risk objective."""
    w = {
        "flow": 1.0, "contact": 1.0, "spatial_collision": 2.0,
        "success": 2.0, "collision": 2.0, "ranking": 0.5,
    }
    if weights:
        w.update(weights)
    point_error = F.smooth_l1_loss(
        output.future_point_flow, future_point_flow, reduction="none"
    ).mean(dim=-1)
    flow_nll = 0.5 * (
        point_error * torch.exp(-output.flow_log_variance) + output.flow_log_variance
    ).mean()
    losses = {
        "flow": flow_nll,
        "flow_smooth_l1": point_error.mean(),
        "contact": F.binary_cross_entropy_with_logits(
            output.task_contact_logits, task_contact_map),
        "spatial_collision": F.binary_cross_entropy_with_logits(
            output.harmful_collision_logits, harmful_collision_map),
        "success": F.binary_cross_entropy_with_logits(output.success_logits, success),
        "collision": F.binary_cross_entropy_with_logits(output.collision_logits, collision),
        "success_rank": _pairwise_rank_loss(output.success_logits, success, True),
        "collision_rank": _pairwise_rank_loss(output.collision_logits, collision, True),
    }
    losses["total"] = (
        w["flow"] * losses["flow"] + w["contact"] * losses["contact"]
        + w["spatial_collision"] * losses["spatial_collision"]
        + w["success"] * losses["success"] + w["collision"] * losses["collision"]
        + w["ranking"] * (losses["success_rank"] + losses["collision_rank"])
    )
    return losses
