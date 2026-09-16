import pytest
import torch

from models.omega_adapter import (
    OmegaAdapterConfig,
    VGGTOmegaSceneAdapter,
    backproject_depth_to_robot,
)


def test_backproject_depth_handles_invalid_values_and_shapes():
    depth = torch.tensor([[[1.0, float("nan")], [2.0, 3.0]]])
    conf = torch.ones_like(depth)
    intr = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0, 0, 1.0]]])
    eye = torch.eye(4)[None]
    points, features = backproject_depth_to_robot(depth, conf, intr, eye, torch.eye(4), 4)
    assert points.shape == (1, 4, 3)
    assert features.shape == (1, 4, 1)
    assert torch.isfinite(points).all()
    assert features[0, 1].item() == 0.0


def test_adapter_rejects_bad_rgb_without_loading_omega():
    adapter = VGGTOmegaSceneAdapter(OmegaAdapterConfig(device="cpu"))
    with pytest.raises(ValueError, match="rgb"):
        adapter.encode(torch.zeros(3, 8, 8), torch.eye(3)[None], torch.eye(4)[None], torch.eye(4))

