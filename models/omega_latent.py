"""Frozen VGGT-Omega latent/register token extraction.

Omega's public model exposes camera/register tokens but not a stable latent
API.  We therefore call the frozen Aggregator directly and select one of its
cached decoder-compatible layers (layer 4 by default).  This module contains
no trainable parameters and remains importable without the Omega checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class OmegaLatentConfig:
    repo: str = "~/VGGT-Omega"
    checkpoint: str = "~/VGGT-Omega/checkpoints/vggt_omega_1b_512.pt"
    resolution: int = 256
    layer_index: int = 4
    device: str = "cuda"


@dataclass
class OmegaLatentOutput:
    tokens: Tensor                 # [B,K,T,D] (D=2*Omega embed dim)
    scene_tokens: Tensor           # [B,K,R,D]
    patch_tokens: Tensor            # [B,K,P,D]
    patch_token_start: int
    metadata: dict[str, Any]


def split_omega_tokens(tokens: Tensor, patch_token_start: int) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 4:
        raise ValueError("tokens must have shape [B,K,T,D]")
    if not 0 < patch_token_start < tokens.shape[2]:
        raise ValueError("patch_token_start must split the token dimension")
    return tokens[:, :, :patch_token_start], tokens[:, :, patch_token_start:]


class OmegaLatentExtractor:
    """Load frozen Omega lazily and extract decoder-compatible latent tokens."""

    def __init__(self, config: OmegaLatentConfig | Mapping[str, Any] | None = None) -> None:
        self.config = (config if isinstance(config, OmegaLatentConfig)
                       else OmegaLatentConfig(**dict(config)) if config is not None
                       else OmegaLatentConfig())
        requested = self.config.device
        self.device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self._loaded = False

    def available(self) -> bool:
        return (Path(self.config.repo).expanduser() / "vggt_omega").is_dir() and Path(
            self.config.checkpoint).expanduser().is_file()

    def load(self) -> "OmegaLatentExtractor":
        if self._loaded:
            return self
        repo = Path(self.config.repo).expanduser().resolve()
        checkpoint = Path(self.config.checkpoint).expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"VGGT-Omega repository not found: {repo}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")
        import sys
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from vggt_omega.models import VGGTOmega
        model = VGGTOmega().to(self.device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self._loaded = True
        return self

    def encode(self, rgb: np.ndarray | Tensor) -> OmegaLatentOutput:
        images = torch.as_tensor(rgb)
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("rgb must have shape [K,3,H,W] or [B,K,3,H,W]")
        images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images.float().div(255.0)
        else:
            images = images.float()
        images = F.interpolate(
            images.flatten(0, 1), size=(self.config.resolution, self.config.resolution),
            mode="bicubic", align_corners=False,
        ).view(images.shape[0], images.shape[1], 3, self.config.resolution, self.config.resolution)
        self.load()
        assert self.model is not None
        aggregator = self.model.aggregator
        with torch.inference_mode():
            amp = self.device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                cached, patch_token_start = aggregator(images)
        if self.config.layer_index >= len(cached) or cached[self.config.layer_index] is None:
            raise ValueError(
                f"Omega layer {self.config.layer_index} is not cached; available layers are "
                f"{[i for i, value in enumerate(cached) if value is not None]}"
            )
        tokens = cached[self.config.layer_index].float()
        scene, patches = split_omega_tokens(tokens, patch_token_start)
        return OmegaLatentOutput(
            tokens=tokens, scene_tokens=scene, patch_tokens=patches,
            patch_token_start=patch_token_start,
            metadata={
                "source": "VGGT-Omega",
                "layer_index": self.config.layer_index,
                "latent_dim": int(tokens.shape[-1]),
                "scene_token_count": int(scene.shape[2]),
                "patch_token_count": int(patches.shape[2]),
                "frozen": True,
            },
        )
