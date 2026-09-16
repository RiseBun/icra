import torch

from models import PointWorld4DModel, pointworld4d_loss


def test_pointworld4d_geometry_shapes_and_backward():
    model = PointWorld4DModel(
        point_feature_dim=1, robot_state_dim=4, hidden_dim=32,
        future_steps=3, layers=1, heads=4,
    )
    points = torch.randn(2, 4, 16, 3)
    features = torch.ones(2, 4, 16, 1)
    robot_state = torch.randn(2, 4, 4)
    robot_flow = torch.randn(2, 5, 6, 8, 3)
    output = model(points, features, robot_state, robot_flow)
    assert output.future_point_flow.shape == (2, 5, 3, 16, 3)
    assert output.flow_log_variance.shape == (2, 5, 3, 16)
    loss = pointworld4d_loss(output, torch.randn_like(output.future_point_flow))["total"]
    loss.backward()
    assert torch.isfinite(loss)
