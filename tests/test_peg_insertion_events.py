import numpy as np

from scripts.peg_insertion_events import detect_event_boundaries


def test_detects_post_grasp_target_and_wrong_target_boundaries():
    target = np.array([0.2, 0.1, 0.05, 0.03, 0.01])
    wrong = np.array([0.3, 0.2, 0.04, 0.08, 0.09])
    grasped = np.array([False, True, True, True, True])
    result = detect_event_boundaries(target, wrong, grasped, target_threshold=0.035)
    assert result == {"grasp_boundary": 1, "target_approach_boundary": 3,
                      "wrong_target_boundary": None}


def test_wrong_target_requires_post_grasp_and_not_target():
    target = np.array([0.2, 0.2, 0.2, 0.2])
    wrong = np.array([0.01, 0.01, 0.04, 0.02])
    grasped = np.array([False, False, True, True])
    result = detect_event_boundaries(target, wrong, grasped, target_threshold=0.035)
    assert result["grasp_boundary"] == 2
    assert result["wrong_target_boundary"] == 3
