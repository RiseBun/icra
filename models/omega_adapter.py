"""RGB/RGB-D adapter for the frozen VGGT-Omega geometry front-end.

The adapter deliberately exposes a small, dependency-isolated contract.  Omega
is imported only when :meth:`load` is called, so data conversion and unit tests
remain runnable without the research checkout or its CUDA dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class OmegaAdapterConfig:
    repo: str = "~/VGGT-Omega"
    checkpoint: str = "~/VGGT-Omega/checkpoints/vggt_omega_1b_512.pt"
    resolution: int = 256
    point_count: int = 512
    min_scale_anchors: int = 16
    device: str = "cuda"


@dataclass
class OmegaSceneOutput:
    points: Tensor                 # [K,N,3] or [B,K,N,3]
    point_features: Tensor         # [K,N,C] or [B,K,N,C]
    confidence: Tensor             # [K,N] or [B,K,N]
    metadata: dict[str, Any] = field(default_factory=dict)


def transform_points(points: Tensor, matrix: Tensor) -> Tensor:
    ones = torch.ones_like(points[..., :1])
    return torch.einsum("...ij,...nj->...ni", matrix,
                        torch.cat((points, ones), dim=-1))[..., :3]


def backproject_depth_to_robot(
    depth: Tensor,
    confidence: Tensor,
    intrinsics: Tensor,
    camera_to_world: Tensor,
    robot_base_to_world: Tensor,
    point_count: int,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Backproject a depth sequence into robot-base coordinates.

    This pure function is also the RGB-D privileged path and is intentionally
    tested independently from Omega.  Invalid depth values produce finite zero
    points with zero confidence rather than poisoning the downstream model.
    """
    if depth.ndim != 3:
        raise ValueError("depth must have shape [K,H,W]")
    if confidence.shape != depth.shape:
        raise ValueError("confidence must have the same shape as depth")
    if intrinsics.shape != (depth.shape[0], 3, 3):
        raise ValueError("intrinsics must have shape [K,3,3]")
    if camera_to_world.shape != (depth.shape[0], 4, 4):
        raise ValueError("camera_to_world must have shape [K,4,4]")
    if robot_base_to_world.shape != (4, 4):
        raise ValueError("robot_base_to_world must have shape [4,4]")
    k, h, w = depth.shape
    ids = torch.linspace(0, h * w - 1, point_count, device=depth.device).long()
    u = (ids % w).float()
    v = torch.div(ids, w, rounding_mode="floor").float()
    world_to_robot = torch.linalg.inv(robot_base_to_world)
    points, features = [], []
    for frame in range(k):
        z_raw = depth[frame].flatten()[ids]
        valid = torch.isfinite(z_raw) & (z_raw > 1e-5)
        z = torch.where(valid, z_raw * float(scale), torch.ones_like(z_raw))
        fx, fy = intrinsics[frame, 0, 0], intrinsics[frame, 1, 1]
        cx, cy = intrinsics[frame, 0, 2], intrinsics[frame, 1, 2]
        camera = torch.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), dim=-1)
        robot = transform_points(transform_points(camera, camera_to_world[frame]),
                                 world_to_robot)
        robot = torch.where(valid[:, None], robot, torch.zeros_like(robot))
        conf = confidence[frame].flatten()[ids].float().clamp(0.0, 1.0)
        conf = torch.where(valid, conf, torch.zeros_like(conf))
        points.append(robot)
        features.append(conf[:, None])
    return torch.stack(points), torch.stack(features)


class VGGTOmegaSceneAdapter:
    """Run frozen Omega and return a robot-base point-token history.

    If Omega latent tokens are not exposed by the installed release, the
    adapter returns depth/confidence tokens and records that downgrade in
    ``metadata['feature_source']``.
    """

    def __init__(self, config: OmegaAdapterConfig | Mapping[str, Any] | None = None) -> None:
        self.config = (config if isinstance(config, OmegaAdapterConfig)
                       else OmegaAdapterConfig(**dict(config)) if config is not None
                       else OmegaAdapterConfig())
        requested = self.config.device
        self.device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self._loaded = False

    def available(self) -> bool:
        return (Path(self.config.repo).expanduser() / "vggt_omega").is_dir() and Path(
            self.config.checkpoint).expanduser().is_file()

    def load(self) -> "VGGTOmegaSceneAdapter":
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

    def encode(
        self,
        rgb: np.ndarray | Tensor,
        intrinsics: np.ndarray | Tensor,
        camera_extrinsics: np.ndarray | Tensor,
        robot_base_to_world: np.ndarray | Tensor,
        *,
        depth: np.ndarray | Tensor | None = None,
        depth_confidence: np.ndarray | Tensor | None = None,
        scale: float | None = None,
        point_count: int | None = None,
    ) -> OmegaSceneOutput:
        rgb_t = torch.as_tensor(rgb)
        if rgb_t.ndim == 5:
            if rgb_t.shape[0] != 1:
                raise ValueError("encode currently accepts one sequence or batch size 1")
            rgb_t = rgb_t[0]
        if rgb_t.ndim != 4 or rgb_t.shape[1] != 3:
            raise ValueError("rgb must have shape [K,3,H,W]")
        intr = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device)
        extr = torch.as_tensor(camera_extrinsics, dtype=torch.float32, device=self.device)
        base = torch.as_tensor(robot_base_to_world, dtype=torch.float32, device=self.device)
        if depth is not None:
            dep = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
            conf = (torch.ones_like(dep) if depth_confidence is None else
                    torch.as_tensor(depth_confidence, dtype=torch.float32, device=self.device))
            points, features = backproject_depth_to_robot(
                dep, conf, intr, extr, base, point_count or self.config.point_count,
                1.0 if scale is None else scale,
            )
            return OmegaSceneOutput(points, features, features[..., 0], {
                "geometry_source": "privileged_rgbd", "feature_source": "depth_confidence",
                "coordinates": "robot_base", "latent_available": False,
            })

        self.load()
        assert self.model is not None
        from scripts.export_omega_risk4d import omega_depth, scaled_intrinsics
        rgb_np = rgb_t.detach().cpu().numpy()
        source_hw = tuple(rgb_t.shape[-2:])
        with torch.inference_mode():
            predicted_depth, confidence, output_hw = omega_depth(
                self.model, rgb_np, self.config.resolution, self.device)
        scaled = scaled_intrinsics(intr.detach().cpu().numpy(), source_hw, output_hw, self.device)
        dep_scale = 1.0 if scale is None else float(scale)
        points, features = backproject_depth_to_robot(
            predicted_depth, confidence, scaled, extr,
            base, point_count or self.config.point_count, dep_scale,
        )
        return OmegaSceneOutput(points, features, features[..., 0], {
            "geometry_source": "vggt_omega", "feature_source": "omega_depth_confidence",
            "coordinates": "robot_base", "latent_available": False,
            "scale": dep_scale, "scale_source": "explicit_or_unity",
        })
