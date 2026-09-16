"""One-step selective controller for action execution or active observation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from baselines.active_view import SelectiveDecision, select_with_observation_budget
from .calibration import RiskCalibration
from .ensemble import Risk4DEnsemble
from .view_value import ViewValueModel, ViewValueOutput


@dataclass
class ControllerOutput:
    success_probability: Tensor
    collision_probability: Tensor
    calibrated_risk_bound: Tensor
    future_point_flow: Tensor
    task_contact_map: Tensor
    harmful_collision_map: Tensor
    view_value: ViewValueOutput
    decision: SelectiveDecision


class RiskAwareController(nn.Module):
    """Compose frozen perception outputs with learned risk and view modules."""

    def __init__(
        self,
        risk_ensemble: Risk4DEnsemble,
        view_model: ViewValueModel,
        calibration: RiskCalibration,
        risk_threshold: float = 0.35,
        max_observations: int = 2,
        failure_weight: float = 1.0,
        collision_weight: float = 1.0,
        confidence_z: float = 1.645,
        move_cost_weight: float = 0.05,
        safety_cost_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.risk_ensemble = risk_ensemble
        self.view_model = view_model
        self.calibration = calibration
        self.risk_threshold = risk_threshold
        self.max_observations = max_observations
        self.failure_weight = failure_weight
        self.collision_weight = collision_weight
        self.confidence_z = confidence_z
        self.move_cost_weight = move_cost_weight
        self.safety_cost_weight = safety_cost_weight

    def forward(
        self,
        points: Tensor,
        point_features: Tensor,
        robot_state: Tensor,
        robot_action_flow: Tensor,
        task_embedding: Tensor,
        candidate_views: Tensor,
        view_move_cost: Tensor,
        view_safety_cost: Tensor,
        observation_count: Tensor,
    ) -> ControllerOutput:
        ensemble = self.risk_ensemble(
            points, point_features, robot_state, robot_action_flow, task_embedding,
        )
        success_logits = torch.stack([member.success_logits for member in ensemble.members]).mean(0)
        collision_logits = torch.stack([member.collision_logits for member in ensemble.members]).mean(0)
        success_probability, collision_probability = self.calibration.probabilities(
            success_logits, collision_logits,
        )
        risk_bound = self.calibration.risk_bound(
            success_logits, collision_logits, ensemble.epistemic_uncertainty,
            self.failure_weight, self.collision_weight, self.confidence_z,
        )
        belief = torch.stack([member.belief for member in ensemble.members]).mean(0)
        view_value = self.view_model(belief, candidate_views, risk_bound)
        decision = select_with_observation_budget(
            risk_bound, view_value.mean_risk_reduction, view_value.std,
            view_move_cost, view_safety_cost, observation_count,
            self.risk_threshold, self.max_observations, self.confidence_z,
            self.move_cost_weight, self.safety_cost_weight,
        )
        return ControllerOutput(
            success_probability=success_probability,
            collision_probability=collision_probability,
            calibrated_risk_bound=risk_bound,
            future_point_flow=torch.stack([
                member.future_point_flow for member in ensemble.members
            ]).mean(0),
            task_contact_map=torch.stack([
                member.task_contact_logits.sigmoid() for member in ensemble.members
            ]).mean(0),
            harmful_collision_map=torch.stack([
                member.harmful_collision_logits.sigmoid() for member in ensemble.members
            ]).mean(0),
            view_value=view_value,
            decision=decision,
        )
