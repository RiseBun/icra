"""Compact action-conditioned transition model in frozen Omega latent space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class OmegaActionWorldOutput:
    future_tokens: Tensor             # [B,M,H,T,D]
    passive_tokens: Tensor            # [B,H,T,D]
    action_residual: Tensor           # [B,M,H,T,D]
    latent_log_variance: Tensor       # [B,M,H,T]


class OmegaActionWorld(nn.Module):
    """Predict future Omega tokens with a shared passive prior and action residual.

    Omega's encoder/decoder is external and frozen.  Only this transition model
    is trainable. Candidate actions share the encoded scene context, while the
    robot-surface flow supplies a spatially meaningful intervention signal.
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        robot_state_dim: int = 16,
        hidden_dim: int = 256,
        future_steps: int = 8,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.latent_dim, self.hidden_dim, self.future_steps = latent_dim, hidden_dim, future_steps
        self.latent_in = nn.Linear(latent_dim, hidden_dim)
        self.state_in = nn.Sequential(nn.Linear(robot_state_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.flow_in = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.scene_temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.action_temporal = nn.TransformerEncoder(layer, num_layers=max(1, layers // 2))
        self.candidate_temporal = nn.TransformerEncoder(layer, num_layers=max(1, layers // 2))
        self.passive_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, future_steps * hidden_dim))
        self.residual_head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, future_steps * hidden_dim))
        self.token_decoder = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, latent_dim))
        self.logvar_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                         nn.Linear(hidden_dim // 2, future_steps))

    def forward(
        self,
        scene_tokens: Tensor,
        robot_state: Tensor,
        robot_action_flow: Tensor,
    ) -> OmegaActionWorldOutput:
        if scene_tokens.ndim != 4:
            raise ValueError("scene_tokens must have shape [B,K,T,D]")
        if robot_state.ndim == 2:
            robot_state = robot_state[:, None]
        if robot_state.ndim != 3 or robot_state.shape[:2] != scene_tokens.shape[:2]:
            raise ValueError("robot_state must have shape [B,K,S]")
        if robot_action_flow.ndim != 5 or robot_action_flow.shape[-1] != 3:
            raise ValueError("robot_action_flow must have shape [B,M,L,Q,3]")
        b, _, tokens, latent_dim = scene_tokens.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"expected latent dim {self.latent_dim}, got {latent_dim}")
        m = robot_action_flow.shape[1]
        scene = self.latent_in(scene_tokens).mean(2)
        scene = self.scene_temporal(scene)[:, -1]
        scene = scene + self.state_in(robot_state[:, -1])
        passive_hidden = self.passive_head(scene).view(b, self.future_steps, self.hidden_dim)
        passive_tokens = self.token_decoder(
            torch.cat((passive_hidden[:, :, None].expand(-1, -1, tokens, -1),
                       self.latent_in(scene_tokens[:, -1])[:, None].expand(-1, self.future_steps, -1, -1)), -1)
        )
        flow_stats = torch.cat((robot_action_flow.amin(3), robot_action_flow.amax(3)), dim=-1)
        action = self.flow_in(flow_stats).view(b * m, flow_stats.shape[2], self.hidden_dim)
        action = self.action_temporal(action).mean(1).view(b, m, self.hidden_dim)
        candidate = self.candidate_temporal(scene[:, None] + action)
        residual_hidden = self.residual_head(
            torch.cat((candidate, scene[:, None].expand(-1, m, -1)), -1)
        ).view(b, m, self.future_steps, self.hidden_dim)
        residual = self.token_decoder(
            torch.cat((residual_hidden[:, :, :, None].expand(-1, -1, -1, tokens, -1),
                       self.latent_in(scene_tokens[:, -1])[:, None, None].expand(-1, m, self.future_steps, -1, -1)), -1)
        )
        future = passive_tokens[:, None] + residual
        logvar = self.logvar_head(candidate).clamp(-8.0, 8.0)
        return OmegaActionWorldOutput(future, passive_tokens, residual, logvar)


def omega_actionworld_loss(output: OmegaActionWorldOutput, target_tokens: Tensor) -> dict[str, Tensor]:
    """Heteroscedastic latent loss; target is [B,M,H,T,D]."""
    if target_tokens.shape != output.future_tokens.shape:
        raise ValueError("target_tokens must match future_tokens")
    passive_target = target_tokens.mean(dim=1)
    residual_target = target_tokens - passive_target[:, None]
    error = F.smooth_l1_loss(output.future_tokens, target_tokens, reduction="none").mean(-1)
    logvar = output.latent_log_variance[:, :, :, None].expand_as(error)
    nll = 0.5 * (error * torch.exp(-logvar) + logvar).mean()
    passive_loss = F.smooth_l1_loss(output.passive_tokens, passive_target)
    residual_loss = F.smooth_l1_loss(output.action_residual, residual_target)
    total = nll + 0.5 * passive_loss + 2.0 * residual_loss
    return {"total": total, "latent_nll": nll,
            "passive_smooth_l1": passive_loss,
            "action_residual_smooth_l1": residual_loss,
            "latent_smooth_l1": error.mean()}
