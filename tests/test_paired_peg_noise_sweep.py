import numpy as np

from scripts.collect_paired_peg_noise_sweep import (
    classify_relation,
    make_paired_noise_actions,
    summarize_relations,
)


def test_paired_actions_reuse_direction_and_match_l2():
    expert = np.zeros((8, 8), np.float32)
    expert[:, 7] = 1.0
    candidates = make_paired_noise_actions(
        expert, 1, 4, [0.02, 0.08], np.random.default_rng(4))
    small = candidates["0.02000"]
    large = candidates["0.08000"]
    for name in ("noise_pos", "noise_neg"):
        np.testing.assert_allclose(np.linalg.norm(small[name][:, :7], axis=-1), 0.02, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(large[name][:, :7], axis=-1), 0.08, atol=1e-6)
    np.testing.assert_allclose(
        large["noise_pos"][:, :7], 4.0 * small["noise_pos"][:, :7], atol=1e-6)
    np.testing.assert_allclose(
        large["noise_neg"][:, :7], 4.0 * small["noise_neg"][:, :7], atol=1e-6)


def test_relation_and_summary_match_fixed_decision_rule():
    assert classify_relation(True, False) == "anti_correlated"
    assert classify_relation(False, True) == "anti_correlated"
    assert classify_relation(True, True) == "both_success"
    assert classify_relation(False, False) == "both_failure"
    records = [{"amplitudes": {
        "0.02000": {"noise_pos": {"success": True}, "noise_neg": {"success": False}},
        "0.04000": {"noise_pos": {"success": False}, "noise_neg": {"success": False}},
    }}, {"amplitudes": {
        "0.02000": {"noise_pos": {"success": True}, "noise_neg": {"success": True}},
        "0.04000": {"noise_pos": {"success": False}, "noise_neg": {"success": False}},
    }}]
    summary = summarize_relations(records)
    assert summary["0.02000"]["anti_correlated"] == 1
    assert summary["0.02000"]["both_success"] == 1
    assert summary["0.04000"]["both_failure"] == 2
