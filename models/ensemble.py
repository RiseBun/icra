"""Deep-ensemble aggregation for calibrated candidate risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .risk4d import Risk4DOutput


@dataclass
class EnsembleRiskOutput:
    members: list[Risk4DOutput]
    success_probability: Tensor
    collision_probability: Tensor
    epistemic_uncertainty: Tensor
    aleatoric_uncertainty: Tensor


class Risk4DEnsemble(nn.Module):
    def __init__(self, members: Iterable[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(list(members))
        if len(self.members) < 1:
            raise ValueError("ensemble requires at least one member")

    def forward(self, *args, **kwargs) -> EnsembleRiskOutput:
        outputs = [member(*args, **kwargs) for member in self.members]
        success = torch.stack([o.success_logits.sigmoid() for o in outputs])
        collision = torch.stack([o.collision_logits.sigmoid() for o in outputs])
        joint = torch.stack((success, collision), dim=-1)
        epistemic = joint.var(dim=0, correction=0).mean(dim=-1)
        aleatoric = torch.stack([o.aleatoric_uncertainty for o in outputs]).mean(dim=0)
        return EnsembleRiskOutput(
            members=outputs,
            success_probability=success.mean(dim=0),
            collision_probability=collision.mean(dim=0),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
        )
