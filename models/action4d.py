"""Small trainable head for action-conditioned 4D affordance prediction.

The visual geometry backbone is deliberately outside this module and is expected
to be frozen. This keeps Omega and VGGT4D interchangeable in experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass
class Action4DOutput:
    future_delta: Tensor       # [B, M, H, N, 3]
    affordance_logits: Tensor  # [B, M, N]
    success_logits: Tensor      # [B, M]
    collision_logits: Tensor    # [B, M]
    uncertainty: Tensor          # [B, M]


class ActionConditioned4DModel(nn.Module):
    """Predict future motion and task risk for candidate action chunks.

    Inputs use fixed-size point samples so the first implementation remains easy
    to benchmark on four 24 GB GPUs. Variable point counts can be supported later
    by passing a mask and replacing mean pooling with masked pooling.
    """

    def __init__(
        self,
        point_feature_dim: int = 0,
        robot_state_dim: int = 16,
        action_dim: int = 10,
        hidden_dim: int = 256,
        future_steps: int = 8,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        relative_actions: bool = False,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.relative_actions = relative_actions
        point_in = 3 + point_feature_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(point_in, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            # Candidates in one clip share the same scene. Encoding the
            # candidate delta from the scene's action centroid makes the head
            # explicitly counterfactual and removes a common shortcut where
            # absolute action magnitude stands in for outcome.
            nn.Linear((2 if relative_actions else 1) * action_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, future_steps * 3),
        )
        self.point_future_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, future_steps * 3),
        )
        self.affordance_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3),  # success, collision, log-variance
        )

    def forward(
        self,
        points: Tensor,
        actions: Tensor,
        robot_state: Tensor,
        features: Optional[Tensor] = None,
    ) -> Action4DOutput:
        """Run the model.

        Args:
            points: [B, K, N, 3]
            actions: [B, M, A]
            robot_state: [B, S]
            features: optional [B, K, N, C]
        """
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("points must have shape [B,K,N,3]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [B,M,A]")
        b, k, n, _ = points.shape
        m = actions.shape[1]
        if features is not None:
            if features.shape[:3] != points.shape[:3]:
                raise ValueError("features must align with points in B,K,N")
            point_input = torch.cat((points, features), dim=-1)
        else:
            point_input = points

        # Pool each frame into one temporal token; retain the last-frame point
        # tokens for point-level affordance and motion decoding.
        point_tokens = self.point_encoder(point_input)
        if features is not None and features.shape[-1] > 0:
            confidence = features[..., :1].detach().clamp(0.05, 1.0)
            frame_tokens = (point_tokens * confidence).sum(dim=2) / confidence.sum(dim=2).clamp_min(1e-4)
        else:
            frame_tokens = point_tokens.mean(dim=2)
        scene_tokens = self.temporal(frame_tokens)
        scene = scene_tokens[:, -1] + self.robot_encoder(robot_state)

        if self.relative_actions:
            action_delta = actions - actions.mean(dim=1, keepdim=True)
            action_input = torch.cat((actions, action_delta), dim=-1)
        else:
            action_input = actions
        action_tokens = self.action_encoder(action_input)
        context = scene[:, None, :].expand(-1, m, -1) + action_tokens
        context = self.temporal(context)
        last_points = self.point_encoder(point_input[:, -1])
        point_context = torch.cat(
            (
                context[:, :, None, :].expand(-1, -1, n, -1),
                last_points[:, None, :, :].expand(-1, m, -1, -1),
            ),
            dim=-1,
        )

        future_delta = self.future_head(context).view(
            b, m, self.future_steps, 3
        )[:, :, :, None, :].expand(-1, -1, -1, n, -1)
        # A point-specific residual makes the future output depend on geometry,
        # while the compact head keeps the first experiments inexpensive.
        point_residual = self.point_future_head(point_context).view(
            b, m, n, self.future_steps, 3
        ).permute(0, 1, 3, 2, 4)
        future_delta = future_delta + 0.1 * point_residual
        affordance = self.affordance_head(point_context).squeeze(-1)
        risk = self.risk_head(context)
        return Action4DOutput(
            future_delta=future_delta,
            affordance_logits=affordance,
            success_logits=risk[..., 0],
            collision_logits=risk[..., 1],
            uncertainty=torch.exp(risk[..., 2].clamp(-8.0, 8.0)),
        )


def action4d_loss(
    output: Action4DOutput,
    future_delta_target: Optional[Tensor] = None,
    affordance_target: Optional[Tensor] = None,
    success_target: Optional[Tensor] = None,
    collision_target: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Return separately logged losses for staged training and ablations."""
    losses: Dict[str, Tensor] = {}
    zero = output.success_logits.sum() * 0.0
    if future_delta_target is not None:
        point_error = torch.nn.functional.smooth_l1_loss(
            output.future_delta, future_delta_target, reduction="none"
        ).mean(dim=(2, 3, 4))
        log_var = output.uncertainty.clamp_min(1e-5).log()
        losses["future"] = (0.5 * (point_error * torch.exp(-log_var) + log_var)).mean()
        losses["future_smooth_l1"] = point_error.mean()
        losses["uncertainty_reg"] = 1e-3 * (log_var ** 2).mean()
    else:
        losses["future"] = zero
        losses["future_smooth_l1"] = zero
        losses["uncertainty_reg"] = zero
    losses["affordance"] = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            output.affordance_logits, affordance_target
        ) if affordance_target is not None else zero
    )
    losses["success"] = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            output.success_logits, success_target
        ) if success_target is not None else zero
    )
    if success_target is not None and success_target.shape[1] > 1:
        success_pair = success_target[:, :, None] - success_target[:, None, :]
        success_mask = success_pair > 0
        success_margin = 0.2 - (output.success_logits[:, :, None] - output.success_logits[:, None, :])
        if success_mask.any():
            losses["success_rank"] = torch.nn.functional.relu(success_margin[success_mask]).mean()
        else:
            losses["success_rank"] = zero
    else:
        losses["success_rank"] = zero
    losses["collision"] = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            output.collision_logits, collision_target
        ) if collision_target is not None else zero
    )
    # BCE calibrates each candidate independently, but active action selection
    # needs the relative ordering of safe and unsafe candidates.  This margin
    # term directly trains the risk head on within-scene pairs and is zero when
    # a clip has no positive/negative contrast.
    if collision_target is not None and collision_target.shape[1] > 1:
        logits = output.collision_logits
        labels = collision_target
        pair_label = labels[:, :, None] - labels[:, None, :]
        pair_mask = pair_label > 0
        pair_margin = 0.2 - (logits[:, :, None] - logits[:, None, :])
        if pair_mask.any():
            losses["collision_rank"] = torch.nn.functional.relu(pair_margin[pair_mask]).mean()
        else:
            losses["collision_rank"] = zero
    else:
        losses["collision_rank"] = zero
    losses["total"] = (
        losses["future"] + losses["uncertainty_reg"] + losses["affordance"]
        + 2.0 * losses["success"] + 1.0 * losses["success_rank"]
        + losses["collision"] + 0.5 * losses["collision_rank"]
    )
    return losses
