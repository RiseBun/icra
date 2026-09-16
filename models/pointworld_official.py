"""PointWorld checkpoint wrapper with an auditable custom fallback.

The official repository has a WDS/data-info contract in addition to the model
weights.  This wrapper keeps that integration optional: when a compatible
loader/checkpoint is supplied it can be selected, otherwise the same public
API uses our compact adapter and reports ``backend='custom_adapter'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import Tensor

from .pointworld4d import PointWorld4DModel, PointWorld4DOutput


@dataclass
class PointWorldBackendInfo:
    backend: str
    checkpoint: str | None
    frozen_backbone: bool
    note: str


class PointWorldDynamicsWrapper:
    """Stable prediction interface for official PointWorld or fallback model."""

    def __init__(
        self,
        *,
        official_checkpoint: str | Path | None = None,
        custom_checkpoint: str | Path | None = None,
        custom_model_config: dict | None = None,
        device: str | torch.device = "cpu",
        official_loader: Optional[Callable[[Path, torch.device], torch.nn.Module]] = None,
        allow_fallback: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.info: PointWorldBackendInfo
        official = Path(official_checkpoint).expanduser() if official_checkpoint else None
        if official is not None and official.is_file() and official_loader is not None:
            try:
                self.model = official_loader(official, self.device).to(self.device).eval()
                for p in self.model.parameters():
                    p.requires_grad_(False)
                self.info = PointWorldBackendInfo(
                    "official", str(official), True,
                    "official loader accepted checkpoint and froze all parameters",
                )
                return
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError("failed to load official PointWorld checkpoint") from exc
                official_note = f"official load failed: {type(exc).__name__}: {exc}"
        elif official is not None and official.is_file():
            official_note = "official checkpoint found but no compatible loader was supplied"
        else:
            official_note = "official PointWorld checkpoint is unavailable"

        if custom_checkpoint is None:
            raise FileNotFoundError(
                f"{official_note}; provide custom_checkpoint for the explicit fallback")
        custom_path = Path(custom_checkpoint).expanduser()
        state = torch.load(custom_path, map_location="cpu", weights_only=True)
        config = custom_model_config or state.get("model_config")
        if config is None:
            raise KeyError("custom checkpoint must contain model_config")
        model = PointWorld4DModel(**config)
        model.load_state_dict(state["model"])
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.info = PointWorldBackendInfo(
            "custom_adapter", str(custom_path), True,
            f"{official_note}; compact PointWorld-like adapter selected",
        )

    @property
    def backend(self) -> str:
        return self.info.backend

    def parameter_counts(self) -> tuple[int, int]:
        if self.model is None:
            return 0, 0
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    @torch.inference_mode()
    def predict(
        self,
        points: Tensor,
        point_features: Tensor,
        robot_state: Tensor,
        robot_action_flow: Tensor,
    ) -> PointWorld4DOutput:
        if self.model is None:
            raise RuntimeError("PointWorld backend has not been loaded")
        return self.model(
            points.to(self.device), point_features.to(self.device),
            robot_state.to(self.device), robot_action_flow.to(self.device),
        )
