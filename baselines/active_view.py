"""Risk-driven action/view selection used by the first closed-loop prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class SelectiveDecision:
    mode: Tensor          # [B], 0=execute, 1=observe, 2=recover
    index: Tensor         # [B], action/view index or -1 for recover
    score: Tensor         # [B], negative action risk or selected view value
    action_risk: Tensor   # [B], minimum calibrated action risk


def select_action_or_view(
    success_logits: Tensor,
    collision_logits: Tensor,
    uncertainty: Tensor,
    view_risk_after: Tensor,
    view_cost: Tensor,
    risk_threshold: float = 0.35,
    camera_cost_weight: float = 0.05,
    collision_weight: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Select action or a view.

    Returns `(mode, index, score)` with mode 0=execute action and 1=observe.
    Inputs are per-batch tensors: action tensors [B,M], view tensors [B,V].
    """
    success = torch.sigmoid(success_logits)
    collision = torch.sigmoid(collision_logits)
    action_score = success - collision_weight * collision - uncertainty
    best_action_score, best_action = action_score.max(dim=-1)
    best_action_risk = 1.0 - success.gather(1, best_action[:, None]).squeeze(1)
    view_score = -view_risk_after - camera_cost_weight * view_cost
    best_view_score, best_view = view_score.max(dim=-1)
    observe = best_action_risk > risk_threshold
    mode = observe.long()
    index = torch.where(observe, best_view, best_action)
    score = torch.where(observe, best_view_score, best_action_score)
    return mode, index, score


def select_with_observation_budget(
    calibrated_risk_bound: Tensor,
    view_risk_reduction: Tensor,
    view_uncertainty: Tensor,
    view_move_cost: Tensor,
    view_safety_cost: Tensor,
    observation_count: Tensor,
    risk_threshold: float = 0.35,
    max_observations: int = 2,
    confidence_z: float = 1.645,
    move_cost_weight: float = 0.05,
    safety_cost_weight: float = 1.0,
    min_view_score: float = 0.0,
) -> SelectiveDecision:
    """Execute if calibrated-safe, otherwise observe only when useful and allowed."""
    if calibrated_risk_bound.ndim != 2:
        raise ValueError("calibrated_risk_bound must have shape [B,M]")
    if view_risk_reduction.shape != view_uncertainty.shape:
        raise ValueError("view mean and uncertainty must have the same [B,V] shape")
    if view_move_cost.shape != view_risk_reduction.shape or view_safety_cost.shape != view_risk_reduction.shape:
        raise ValueError("view costs must align with [B,V]")
    if observation_count.ndim != 1 or observation_count.shape[0] != calibrated_risk_bound.shape[0]:
        raise ValueError("observation_count must have shape [B]")

    best_action_risk, best_action = calibrated_risk_bound.min(dim=-1)
    view_lcb = view_risk_reduction - confidence_z * view_uncertainty
    view_score = view_lcb - move_cost_weight * view_move_cost - safety_cost_weight * view_safety_cost
    best_view_score, best_view = view_score.max(dim=-1)

    execute = best_action_risk <= risk_threshold
    can_observe = (observation_count < max_observations) & (best_view_score > min_view_score)
    observe = (~execute) & can_observe
    recover = (~execute) & (~can_observe)
    mode = torch.full_like(best_action, 2)
    mode[execute] = 0
    mode[observe] = 1
    index = torch.full_like(best_action, -1)
    index[execute] = best_action[execute]
    index[observe] = best_view[observe]
    score = torch.zeros_like(best_action_risk)
    score[execute] = -best_action_risk[execute]
    score[observe] = best_view_score[observe]
    score[recover] = -best_action_risk[recover]
    return SelectiveDecision(mode, index, score, best_action_risk)
