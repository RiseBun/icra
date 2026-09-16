import numpy as np

from scripts.collect_rlbench_counterfactuals import pack_handle_sequences, pack_scalar_sequences
from scripts.feasibility_contact_modes import derive_modes


def _sample(m=2, h=3, n=4):
    return {
        "task_contact_map": np.zeros((m, h, n), np.float32),
        "harmful_collision_map": np.zeros((m, h, n), np.float32),
        "future_point_flow": np.zeros((m, h, n, 3), np.float32),
        "success": np.zeros((m,), np.float32),
    }


def test_contact_modes_have_candidate_time_shape_and_precedence():
    sample = _sample()
    sample["future_point_flow"][0, 0, :, 0] = 0.01
    sample["task_contact_map"][0, 1, 0] = 1.0
    sample["future_point_flow"][0, 1, 0, 0] = 0.01
    sample["harmful_collision_map"][1, 0, 0] = 1.0
    sample["success"][0] = 1.0
    modes = derive_modes(sample, motion_threshold=1e-3)
    assert modes.shape == (2, 3)
    assert modes[0, 0] == 1  # moving without contact = approach
    assert modes[0, 1] == 3  # contact and motion = sliding/grasping
    assert modes[1, 0] == 5  # harmful collision has highest precedence
    assert modes[0, 2] == 4  # completion is applied at the final horizon step


def test_raw_contact_signals_are_packed_without_ragged_arrays():
    handles, counts = pack_handle_sequences([[{4, 2}, set()], [{9}]], length=3, max_contacts=2)
    assert handles.shape == (2, 3, 2)
    assert counts.tolist() == [[2, 0, 0], [1, 1, 1]]
    assert handles[0, 0].tolist() == [2, 4]
    progress = pack_scalar_sequences([[0.0, 0.5], [1.0]], length=3)
    np.testing.assert_allclose(progress, [[0.0, 0.5, 0.5], [1.0, 1.0, 1.0]])
