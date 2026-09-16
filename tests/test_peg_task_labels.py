import numpy as np

from scripts.peg_task_labels import (
    derive_grasp_event_modes,
    derive_peg_labels,
    has_grasp_outcome_variation,
)


def test_peg_labels_separate_modes_and_failures():
    shape = (2, 5)
    grasped = np.zeros(shape, bool)
    grasped[0, 2:] = True
    grasped[1, 1] = True
    target = np.full(shape, 0.20, np.float32)
    target[0, 2:] = 0.02
    target[1, 1] = 0.02
    wrong = np.full(shape, 0.20, np.float32)
    wrong[1, 3] = 0.01
    depth = np.zeros(shape, np.float32)
    depth[:, 2:] = 0.02
    align = np.zeros(shape, np.float32)
    align[0, 3] = 0.8
    success = np.asarray([True, False])
    collision = np.zeros(shape, bool)
    collision[1, 4] = True

    mode, failure = derive_peg_labels(
        grasped, target, wrong, depth, align, success, collision,
        approach_distance=0.25)
    assert mode.shape == failure.shape == shape
    assert mode[0, 0] == 1       # approach
    assert mode[0, 2] == 4       # alignment
    assert mode[0, -1] == 5      # completion
    assert failure[1, 2] == 1    # dropped after a prior grasp
    assert failure[1, 3] == 2    # wrong target
    assert failure[1, 4] == 4    # collision has explicit auxiliary label


def test_grasp_event_modes_and_gate():
    grasped = np.asarray([[False, True, True, False], [False, False, False, False]])
    success = np.asarray([True, False])
    modes = derive_grasp_event_modes(grasped, success)
    assert modes.tolist() == [[0, 1, 1, 3], [0, 0, 0, 0]]
    assert has_grasp_outcome_variation(success, np.asarray([False, True]))
    assert not has_grasp_outcome_variation(np.ones(2), np.zeros(2))
