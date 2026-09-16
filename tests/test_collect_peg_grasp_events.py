import numpy as np

from scripts.collect_peg_grasp_events import make_candidates, _pad


def test_candidate_set_has_six_matched_actions():
    expert = np.zeros((10, 8), np.float32)
    expert[:, 7] = 1.0
    candidates = make_candidates(expert, 2, 4, 0.05, np.random.default_rng(0))
    assert [name for name, _ in candidates] == [
        "nominal", "noise_pos", "noise_neg", "gripper_flip", "gripper_hold", "gripper_delay"
    ]
    assert all(action.shape == (4, 8) for _, action in candidates)
    assert all(np.isin(action[:, 7], [0.0, 1.0]).all() for _, action in candidates)

    l2_candidates = make_candidates(expert, 2, 4, 0.05, np.random.default_rng(0), noise_l2=0.04)
    for name in ("noise_pos", "noise_neg"):
        action = dict(l2_candidates)[name]
        np.testing.assert_allclose(np.linalg.norm(action[:, :7] - expert[2:6, :7], axis=-1),
                                   0.04, atol=1e-6)


def test_padding_repeats_last_observation():
    assert _pad([True, False], 4) == [True, False, False, False]
    assert _pad([], 4) == []
