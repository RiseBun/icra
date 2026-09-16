"""Fast CPU/GPU smoke test for the trainable action-conditioned head."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel, action4d_loss
from baselines.constant_velocity import predict_constant_velocity
from baselines.active_view import select_action_or_view


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, k, n, m, h = 2, 8, 64, 8, 8
    model = ActionConditioned4DModel(
        robot_state_dim=16, action_dim=10, future_steps=h,
        hidden_dim=128, layers=2, heads=4,
    ).to(device)
    points = torch.randn(b, k, n, 3, device=device)
    actions = torch.randn(b, m, 10, device=device)
    state = torch.randn(b, 16, device=device)
    out = model(points, actions, state)
    targets = {
        "future_delta_target": torch.randn_like(out.future_delta),
        "affordance_target": torch.rand_like(out.affordance_logits),
        "success_target": torch.rand_like(out.success_logits),
        "collision_target": torch.rand_like(out.collision_logits),
    }
    loss = action4d_loss(out, **targets)["total"]
    loss.backward()
    predicted = predict_constant_velocity(points, h)
    mode, index, score = select_action_or_view(
        out.success_logits.detach(), out.collision_logits.detach(),
        out.uncertainty.detach(), torch.rand(b, 4, device=device),
        torch.rand(b, 4, device=device),
    )
    assert out.future_delta.shape == (b, m, h, n, 3)
    assert predicted.shape == (b, h, n, 3)
    assert mode.shape == index.shape == score.shape == (b,)
    print({
        "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss.detach()), "future": tuple(out.future_delta.shape),
    })


if __name__ == "__main__":
    main()
