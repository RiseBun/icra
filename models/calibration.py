"""Temperature and split-conformal calibration for selective execution."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F


def _finite_sample_quantile(values: Tensor, coverage: float) -> Tensor:
    if values.numel() == 0:
        raise ValueError("calibration split is empty")
    rank = min(values.numel(), math.ceil((values.numel() + 1) * coverage))
    return values.flatten().sort().values[rank - 1]


def fit_temperature(logits: Tensor, labels: Tensor, max_iter: int = 100) -> float:
    """Fit one positive scalar temperature without changing ranking."""
    logits = logits.detach().float().flatten()
    labels = labels.detach().float().flatten()
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


@dataclass
class RiskCalibration:
    success_temperature: float = 1.0
    collision_temperature: float = 1.0
    failure_residual_quantile: float = 0.0
    collision_residual_quantile: float = 0.0
    coverage: float = 0.9

    @classmethod
    def fit(
        cls,
        success_logits: Tensor,
        collision_logits: Tensor,
        success_labels: Tensor,
        collision_labels: Tensor,
        coverage: float = 0.9,
    ) -> "RiskCalibration":
        if not 0.5 < coverage < 1.0:
            raise ValueError("coverage must be between 0.5 and 1.0")
        for name, labels in (("success", success_labels), ("collision", collision_labels)):
            if torch.unique(labels).numel() < 2:
                raise ValueError(
                    f"calibration {name} labels require both classes; collect more episodes"
                )
        success_temperature = fit_temperature(success_logits, success_labels)
        collision_temperature = fit_temperature(collision_logits, collision_labels)
        success_prob = torch.sigmoid(success_logits / success_temperature)
        collision_prob = torch.sigmoid(collision_logits / collision_temperature)
        failure_residual = (1.0 - success_labels) - (1.0 - success_prob)
        collision_residual = collision_labels - collision_prob
        return cls(
            success_temperature=success_temperature,
            collision_temperature=collision_temperature,
            failure_residual_quantile=max(
                0.0, float(_finite_sample_quantile(failure_residual, coverage))),
            collision_residual_quantile=max(
                0.0, float(_finite_sample_quantile(collision_residual, coverage))),
            coverage=coverage,
        )

    def probabilities(self, success_logits: Tensor, collision_logits: Tensor) -> tuple[Tensor, Tensor]:
        return (
            torch.sigmoid(success_logits / self.success_temperature),
            torch.sigmoid(collision_logits / self.collision_temperature),
        )

    def risk_bound(
        self,
        success_logits: Tensor,
        collision_logits: Tensor,
        epistemic_uncertainty: Tensor | None = None,
        failure_weight: float = 1.0,
        collision_weight: float = 1.0,
        confidence_z: float = 1.645,
    ) -> Tensor:
        success_prob, collision_prob = self.probabilities(success_logits, collision_logits)
        epistemic_margin: Tensor | float = 0.0
        if epistemic_uncertainty is not None:
            epistemic_margin = confidence_z * epistemic_uncertainty.clamp_min(0).sqrt()
        failure_ucb = (1.0 - success_prob + self.failure_residual_quantile + epistemic_margin).clamp(0, 1)
        collision_ucb = (collision_prob + self.collision_residual_quantile + epistemic_margin).clamp(0, 1)
        return failure_weight * failure_ucb + collision_weight * collision_ucb

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RiskCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
