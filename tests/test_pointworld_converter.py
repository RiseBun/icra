import numpy as np

from scripts.convert_rlbench_to_pointworld import convert_sample


def test_converter_preserves_labels_and_adds_contract_aliases():
    sample = {
        "points": np.zeros((2, 4, 3), np.float32),
        "point_features": np.ones((2, 4, 1), np.float32),
        "robot_action_flow": np.zeros((3, 5, 6, 3), np.float32),
        "future_point_flow": np.zeros((3, 2, 4, 3), np.float32),
        "success": np.array([1, 0, 1], np.float32),
    }
    out = convert_sample(sample)
    assert np.array_equal(out["success"], sample["success"])
    assert out["scene_points"].shape == sample["points"].shape
    assert out["future_scene_flow"].shape == sample["future_point_flow"].shape
    assert out["robot_points_available"].item() is False
