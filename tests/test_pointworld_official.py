import torch

from models.pointworld4d import PointWorld4DModel
from models.pointworld_official import PointWorldDynamicsWrapper


def test_custom_adapter_fallback_is_explicit(tmp_path):
    config = dict(point_feature_dim=1, robot_state_dim=4, hidden_dim=16,
                  future_steps=2, layers=1, heads=2, dropout=0.0)
    model = PointWorld4DModel(**config)
    ckpt = tmp_path / "custom.pt"
    torch.save({"model": model.state_dict(), "model_config": config}, ckpt)
    wrapper = PointWorldDynamicsWrapper(
        official_checkpoint=tmp_path / "missing.pt", custom_checkpoint=ckpt, device="cpu")
    assert wrapper.backend == "custom_adapter"
    total, trainable = wrapper.parameter_counts()
    assert total > 0 and trainable == 0
    out = wrapper.predict(torch.zeros(1, 2, 4, 3), torch.ones(1, 2, 4, 1),
                          torch.zeros(1, 2, 4), torch.zeros(1, 1, 3, 2, 3))
    assert out.future_point_flow.shape == (1, 1, 2, 4, 3)
