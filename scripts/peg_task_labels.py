"""Pure numpy labels for task outcomes in ``insert_onto_square_peg``.

The labels deliberately separate interaction state from safety/task failure.
They consume simulator-side signals during dataset construction; the RGB-only
model never receives these privileged fields at test time.
"""

from __future__ import annotations

import numpy as np


MODE_NAMES = ("free", "approach", "grasp", "transport", "alignment", "completion")
FAILURE_NAMES = ("none", "drop", "wrong_target", "misalignment", "physical_collision")
GRASP_EVENT_MODE_NAMES = ("approach", "grasp", "drop", "completion")


def derive_grasp_event_modes(grasped: np.ndarray, success: np.ndarray) -> np.ndarray:
    """Map raw grasp observations to compact per-candidate event modes."""
    grasped = np.asarray(grasped).astype(bool)
    if grasped.ndim != 2:
        raise ValueError("grasped must have shape [M, H]")
    success = np.asarray(success).reshape(-1).astype(bool)
    if success.shape != (grasped.shape[0],):
        raise ValueError("success must have shape [M]")
    modes = np.zeros(grasped.shape, dtype=np.int64)
    modes[grasped] = 1
    ever_grasped = np.maximum.accumulate(grasped, axis=1)
    modes[ever_grasped & ~grasped] = 2
    for candidate, complete in enumerate(success):
        if complete:
            modes[candidate, -1] = 3
    return modes


def has_grasp_outcome_variation(success: np.ndarray, grasp_failure: np.ndarray) -> bool:
    """Gate a decision point only when candidate outcomes actually differ."""
    success = np.asarray(success).reshape(-1).astype(bool)
    grasp_failure = np.asarray(grasp_failure).reshape(-1).astype(bool)
    if success.shape != grasp_failure.shape:
        raise ValueError("success and grasp_failure must share shape [M]")
    return bool(np.unique(success).size > 1 or np.unique(grasp_failure).size > 1)


def derive_peg_labels(
    grasped: np.ndarray,
    target_distance: np.ndarray,
    wrong_target_distance: np.ndarray,
    insertion_depth: np.ndarray,
    alignment_error: np.ndarray,
    success: np.ndarray,
    physical_collision: np.ndarray | None = None,
    *,
    approach_distance: float = 0.08,
    target_distance_threshold: float = 0.035,
    insertion_depth_threshold: float = 0.01,
    alignment_threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive per-candidate/time interaction and failure labels.

    All signal arrays have shape ``[M, H]`` except ``success`` with shape
    ``[M]``.  Priority is completion > physical collision > drop/wrong target
    > alignment > grasp > transport > approach > free.  A collision remains
    an auxiliary failure reason even when it occurs during a task-required
    interaction; callers should filter normal contacts before passing it.
    """
    arrays = [np.asarray(value) for value in
              (grasped, target_distance, wrong_target_distance, insertion_depth,
               alignment_error)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("per-step peg signals must all have shape [M, H]")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("per-step peg signals must share one shape")
    m, h = arrays[0].shape
    success = np.asarray(success).reshape(-1).astype(bool)
    if success.shape != (m,):
        raise ValueError("success must have shape [M]")
    if physical_collision is None:
        physical_collision = np.zeros((m, h), dtype=bool)
    physical_collision = np.asarray(physical_collision).astype(bool)
    if physical_collision.shape != (m, h):
        raise ValueError("physical_collision must have shape [M, H]")

    grasped = np.asarray(grasped).astype(bool)
    target_distance = np.asarray(target_distance, dtype=np.float32)
    wrong_target_distance = np.asarray(wrong_target_distance, dtype=np.float32)
    insertion_depth = np.asarray(insertion_depth, dtype=np.float32)
    alignment_error = np.asarray(alignment_error, dtype=np.float32)

    mode = np.full((m, h), 0, dtype=np.int64)  # free
    near_target = target_distance <= target_distance_threshold
    approaching = (~grasped) & (target_distance <= approach_distance)
    mode[approaching] = 1
    mode[grasped] = 2  # grasp; refined below
    mode[grasped & ~near_target] = 3  # transport
    mode[grasped & near_target & (alignment_error <= alignment_threshold)] = 4
    for candidate in range(m):
        if success[candidate]:
            mode[candidate, -1] = 5

    failure = np.full((m, h), 0, dtype=np.int64)  # none
    ever_grasped = np.maximum.accumulate(grasped, axis=1)
    dropped = ever_grasped & ~grasped
    failure[dropped] = 1
    wrong_target = ((wrong_target_distance <= target_distance_threshold)
                    & (target_distance > target_distance_threshold))
    failure[wrong_target] = 2
    misalignment = (grasped & (insertion_depth >= insertion_depth_threshold)
                    & (alignment_error > alignment_threshold))
    failure[misalignment] = 3
    # Preserve collision as the explicit safety label when multiple failure
    # symptoms occur in one frame.
    failure[physical_collision] = 4
    for candidate in range(m):
        if success[candidate]:
            failure[candidate, -1] = 0
    return mode, failure
