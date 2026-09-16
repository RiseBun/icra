"""Strict loader for episode-split risk-conditioned 4D clips."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = (
    "points", "point_features", "robot_state", "robot_action_flow",
    "task_embedding", "future_point_flow", "task_contact_map",
    "harmful_collision_map", "success", "collision",
)
OPTIONAL_KEYS = (
    "candidate_actions", "candidate_views", "view_cost",
    "view_safety_cost", "view_risk_reduction", "view_points",
    "view_point_features",
)


def validate_risk4d_sample(sample: Mapping[str, np.ndarray | torch.Tensor]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in sample]
    if missing:
        raise ValueError(f"risk4d v2 sample is missing real labels/features: {missing}")
    shape = {key: tuple(sample[key].shape) for key in REQUIRED_KEYS}
    if len(shape["points"]) != 3 or shape["points"][-1] != 3:
        raise ValueError("points must have shape [K,N,3]")
    if shape["point_features"][:2] != shape["points"][:2]:
        raise ValueError("point_features must align with points in K,N")
    if len(shape["robot_state"]) != 2 or shape["robot_state"][0] != shape["points"][0]:
        raise ValueError("robot_state must have shape [K,S]")
    if len(shape["robot_action_flow"]) != 4 or shape["robot_action_flow"][-1] != 3:
        raise ValueError("robot_action_flow must have shape [M,L,Q,3]")
    m = shape["robot_action_flow"][0]
    if len(shape["future_point_flow"]) != 4 or shape["future_point_flow"][0] != m:
        raise ValueError("future_point_flow must have shape [M,H,N,3]")
    if shape["future_point_flow"][2:] != (shape["points"][1], 3):
        raise ValueError("future point flow must use the same N as points")
    map_shape = shape["future_point_flow"][:-1]
    if shape["task_contact_map"] != map_shape or shape["harmful_collision_map"] != map_shape:
        raise ValueError("contact and harmful collision maps must have shape [M,H,N]")
    if shape["success"] != (m,) or shape["collision"] != (m,):
        raise ValueError("success and collision must have shape [M]")
    if len(shape["task_embedding"]) != 1:
        raise ValueError("task_embedding must have shape [D]")


class Risk4DDataset(Dataset):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.files = sorted(self.root.glob("**/*.npz"))
        if not self.files:
            raise ValueError(f"no NPZ clips found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.files[index], allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        validate_risk4d_sample(arrays)
        keys = [key for key in REQUIRED_KEYS + OPTIONAL_KEYS if key in arrays]
        return {key: torch.from_numpy(arrays[key]).float() for key in keys}
