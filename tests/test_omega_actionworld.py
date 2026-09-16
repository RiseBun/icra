import torch

from models.omega_actionworld import OmegaActionWorld, omega_actionworld_loss
from models.omega_latent import split_omega_tokens


def test_split_omega_tokens_and_action_world_shapes():
    tokens = torch.randn(2, 8, 21, 32)
    scene, patches = split_omega_tokens(tokens, 5)
    assert scene.shape == (2, 8, 5, 32)
    assert patches.shape == (2, 8, 16, 32)
    model = OmegaActionWorld(latent_dim=32, robot_state_dim=6, hidden_dim=32,
                             future_steps=3, layers=1, heads=4, dropout=0.0)
    state = torch.randn(2, 8, 6)
    robot_flow = torch.randn(2, 4, 5, 7, 3)
    output = model(tokens, state, robot_flow)
    assert output.future_tokens.shape == (2, 4, 3, 21, 32)
    loss = omega_actionworld_loss(output, torch.randn_like(output.future_tokens))["total"]
    loss.backward()
    assert torch.isfinite(loss)
