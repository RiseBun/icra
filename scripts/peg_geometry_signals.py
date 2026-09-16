"""Insertion-layer geometry signals for ``insert_onto_square_peg``.

These are privileged simulator-side signals used only during dataset
construction.  The RGB-only model never receives them at test time.

Signals (all in world frame):
  target_distance       ring -> target pillar horizontal distance
  wrong_target_distance ring -> nearest NON-target pillar horizontal distance
  ring_height_above_pillar  ring z minus target pillar centre z (insertion proxy)
  target_pillar_index   which of pillar0/1/2 is the target
"""

from __future__ import annotations

import numpy as np


def extract_peg_signals(task) -> dict[str, float]:
    """Extract insertion signals from the live simulator state.

    The target pillar is not stored by RLBench as an attribute; it is recovered
    as the pillar whose (x,y) is closest to ``success_centre`` (which RLBench
    moves onto the chosen pillar during ``init_episode``).
    """
    from pyrep.objects.shape import Shape

    inner = getattr(task, "_task", task)
    ring_pos = np.asarray(inner._square_ring.get_position(), dtype=np.float32)
    success_pos = np.asarray(inner._success_centre.get_position(), dtype=np.float32)

    pillars = [Shape("pillar0"), Shape("pillar1"), Shape("pillar2")]
    pillar_positions = np.stack([np.asarray(p.get_position(), dtype=np.float32)
                                 for p in pillars])

    target_xy = success_pos[:2]
    horizontal = np.linalg.norm(pillar_positions[:, :2] - target_xy, axis=1)
    target_idx = int(np.argmin(horizontal))
    target_pillar_pos = pillar_positions[target_idx]

    target_distance = float(np.linalg.norm(ring_pos[:2] - success_pos[:2]))
    non_target = [i for i in range(3) if i != target_idx]
    wrong_target_distance = float(min(
        np.linalg.norm(ring_pos[:2] - pillar_positions[i][:2]) for i in non_target))
    # "insertion" is the ring descending onto the pillar; the natural reference
    # is the target pillar centre, not success_centre (whose z is near the base).
    ring_height_above_pillar = float(ring_pos[2] - target_pillar_pos[2])

    return {
        "target_distance": target_distance,
        "wrong_target_distance": wrong_target_distance,
        "ring_height_above_pillar": ring_height_above_pillar,
        "target_pillar_index": target_idx,
    }


def extract_target_pillar_geometry(task, target_pillar_index: int) -> dict[str, float]:
    """Extract labels from the actual target pillar, never ``success_centre``.

    ``target_pillar_index`` is supplied by the geometry-construction code.  It
    must not be inferred from a Dummy or proximity sensor because those objects
    may retain the original RLBench target after a geometry swap.
    """
    from pyrep.objects.shape import Shape

    if int(target_pillar_index) not in (0, 1, 2):
        raise ValueError("target_pillar_index must be 0, 1, or 2")
    inner = getattr(task, "_task", task)
    ring = np.asarray(inner._square_ring.get_position(), dtype=np.float32)
    pillars = np.stack([
        np.asarray(Shape(f"pillar{i}").get_position(), dtype=np.float32)
        for i in range(3)
    ])
    distances = np.linalg.norm(pillars[:, :2] - ring[:2], axis=1)
    wrong = [i for i in range(3) if i != int(target_pillar_index)]
    nearest = int(np.argmin(distances))
    return {
        "target_pillar_index": int(target_pillar_index),
        "nearest_pillar_index": nearest,
        "target_distance": float(distances[int(target_pillar_index)]),
        "wrong_target_distance": float(min(distances[i] for i in wrong)),
        "ring_height_above_target": float(ring[2] - pillars[int(target_pillar_index), 2]),
        "ring_x": float(ring[0]), "ring_y": float(ring[1]), "ring_z": float(ring[2]),
        "target_x": float(pillars[int(target_pillar_index), 0]),
        "target_y": float(pillars[int(target_pillar_index), 1]),
        "target_z": float(pillars[int(target_pillar_index), 2]),
    }
