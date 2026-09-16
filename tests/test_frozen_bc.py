import numpy as np
import pytest
import torch

from policies.candidate_provider import load_candidate_provider, validate_policy_candidates
from policies.frozen_bc import FrozenBCPolicy


def _checkpoint(path):
    model = FrozenBCPolicy(history_steps=2, state_dim=5, action_horizon=3,
                           image_size=16, hidden_dim=32)
    torch.save({
        "model": model.state_dict(),
        "model_config": {
            "history_steps": 2, "state_dim": 5, "action_horizon": 3,
            "image_size": 16, "hidden_dim": 32,
        },
        "action_semantics": "absolute_joint_position",
    }, path)


def _context():
    return {
        "rgb_history": np.zeros((2, 16, 16, 3), np.uint8),
        "robot_state": np.zeros((2, 5), np.float32),
        "joint_history": np.zeros((2, 8), np.float32),
    }


def test_frozen_bc_provider_is_seeded_and_has_contract(tmp_path):
    checkpoint = tmp_path / "bc.pt"
    _checkpoint(checkpoint)
    provider = load_candidate_provider(
        f"policies.frozen_bc:provide?checkpoint={checkpoint}&device=cpu&temperature=0.5")
    first = provider(_context(), 4, 3, 7)
    second = provider(_context(), 4, 3, 7)
    np.testing.assert_allclose(first, second)
    validate_policy_candidates(first, 4, 3)
    assert set(np.unique(first[..., 7])).issubset({0.0, 1.0})


def test_provider_accepts_shell_safe_semicolon_query(tmp_path):
    checkpoint = tmp_path / "bc.pt"
    _checkpoint(checkpoint)
    provider = load_candidate_provider(
        f"policies.frozen_bc:provide?checkpoint={checkpoint};device=cpu;temperature=0.5")
    actions = provider(_context(), 2, 3, 7)
    assert actions.shape == (2, 3, 8)


def test_frozen_bc_rejects_wrong_horizon(tmp_path):
    checkpoint = tmp_path / "bc.pt"
    _checkpoint(checkpoint)
    provider = load_candidate_provider(
        f"policies.frozen_bc:provide?checkpoint={checkpoint}&device=cpu")
    with pytest.raises(ValueError, match="horizon"):
        provider(_context(), 4, 4, 7)
