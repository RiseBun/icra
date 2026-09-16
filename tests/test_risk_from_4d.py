import torch

from models import RiskFrom4DBeliefModel, risk_from_4d_loss


def test_risk_from_frozen_4d_shapes_and_loss():
    model = RiskFrom4DBeliefModel(
        point_feature_dim=1, robot_state_dim=4, hidden_dim=32,
        future_steps=3, layers=1, heads=4,
    )
    points = torch.randn(2, 4, 16, 3)
    features = torch.ones(2, 4, 16, 1)
    state = torch.randn(2, 4, 4)
    robot_flow = torch.randn(2, 5, 6, 8, 3)
    future = torch.randn(2, 5, 3, 16, 3)
    output = model(points, features, state, robot_flow, future)
    assert output.task_contact_logits.shape == (2, 5, 3, 16)
    assert output.harmful_collision_logits.shape == (2, 5, 3, 16)
    assert output.success_logits.shape == (2, 5)
    loss = risk_from_4d_loss(
        output, torch.rand(2, 5, 3, 16), torch.rand(2, 5, 3, 16),
        torch.rand(2, 5), torch.rand(2, 5),
    )["total"]
    loss.backward()
    assert torch.isfinite(loss)
