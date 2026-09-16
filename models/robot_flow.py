"""Convert joint-space action candidates to robot-surface 3D flow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor


class PointWorldRobotFlowAdapter:
    """Dependency-isolated wrapper around PointWorld's RobotSampler.

    The returned tensor is displacement from the robot surface at the current
    state, not absolute robot points. PointWorld is imported lazily so the core
    risk model and its training environment do not depend on its URDF stack.
    """

    def __init__(
        self,
        pointworld_root: str | Path,
        urdf_path: str | Path,
        num_points: int = 128,
        device: str = "cuda",
        gripper_only: bool = False,
        seed: int = 2027,
    ) -> None:
        root = Path(pointworld_root).expanduser().resolve()
        if not (root / "robot_sampler.py").is_file():
            raise FileNotFoundError(f"PointWorld robot_sampler.py not found in {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from robot_sampler import RobotSampler, convert_joints_to_dict
        except ImportError as exc:
            raise ImportError(
                "PointWorld robot dependencies are unavailable. Use its environment "
                "only for offline robot-flow export."
            ) from exc

        self._convert_joints_to_dict = convert_joints_to_dict
        self.sampler = RobotSampler(
            str(Path(urdf_path).expanduser().resolve()),
            gripper_only=gripper_only,
            device=device,
        )
        self.sampler.presample(num_points, gripper_filter="both", seed=seed)
        self.device = torch.device(device)
        self.num_points = num_points

    @property
    def joint_names(self) -> list[str]:
        return list(self.sampler.joint_names)

    def encode(
        self,
        joint_trajectory: Tensor,
        current_joints: Tensor,
        joint_names: Optional[Sequence[str]] = None,
    ) -> Tensor:
        """Return action-conditioned robot flow with shape [B,M,L,Q,3].

        `joint_trajectory` contains absolute joint positions [B,M,L,J], and
        `current_joints` is the common pre-intervention state [B,J]. Candidate
        rollouts must share this state for the comparison to be causal.
        """
        if joint_trajectory.ndim != 4:
            raise ValueError("joint_trajectory must have shape [B,M,L,J]")
        if current_joints.ndim != 2:
            raise ValueError("current_joints must have shape [B,J]")
        if joint_trajectory.shape[0] != current_joints.shape[0]:
            raise ValueError("trajectory and current joints must have the same batch size")
        names = list(joint_names) if joint_names is not None else self.joint_names
        if joint_trajectory.shape[-1] != len(names) or current_joints.shape[-1] != len(names):
            raise ValueError("last joint dimension must match joint_names")

        trajectory = joint_trajectory.to(self.device, dtype=torch.float32)
        current = current_joints.to(self.device, dtype=torch.float32)
        b, m, steps, _ = trajectory.shape
        trajectory_dict, _ = self._convert_joints_to_dict(trajectory, names)
        current_dict, _ = self._convert_joints_to_dict(current, names)
        trajectory_points, _, _ = self.sampler.compute_points(trajectory_dict)
        current_points, _, _ = self.sampler.compute_points(current_dict)
        trajectory_points = trajectory_points.reshape(b, m, steps, self.num_points, 3)
        return trajectory_points - current_points[:, None, None]

    def surface_points(
        self,
        joint_state: Tensor,
        joint_names: Optional[Sequence[str]] = None,
    ) -> Tensor:
        """Return absolute robot-surface points [B,Q,3] in the URDF base frame."""
        if joint_state.ndim != 2:
            raise ValueError("joint_state must have shape [B,J]")
        names = list(joint_names) if joint_names is not None else self.joint_names
        if joint_state.shape[-1] != len(names):
            raise ValueError("last joint dimension must match joint_names")
        joint_dict, _ = self._convert_joints_to_dict(
            joint_state.to(self.device, dtype=torch.float32), names)
        points, _, _ = self.sampler.compute_points(joint_dict)
        return points


def rlbench_franka_joint_trajectory(
    candidate_actions: Tensor,
    arm_joint_names: Sequence[str],
    finger_joint_names: Sequence[str] = (),
    gripper_open_position: float = 0.04,
) -> tuple[Tensor, list[str]]:
    """Map RLBench's arm joints plus open/close scalar to URDF joints."""
    if candidate_actions.ndim != 4 or candidate_actions.shape[-1] < len(arm_joint_names):
        raise ValueError("candidate_actions must have shape [B,M,L,A]")
    arm = candidate_actions[..., :len(arm_joint_names)]
    names = list(arm_joint_names)
    if not finger_joint_names:
        return arm, names
    if candidate_actions.shape[-1] <= len(arm_joint_names):
        raise ValueError("a gripper scalar is required when finger joints are requested")
    gripper = candidate_actions[..., len(arm_joint_names):len(arm_joint_names) + 1]
    gripper = gripper.clamp(0.0, 1.0) * gripper_open_position
    fingers = gripper.expand(*gripper.shape[:-1], len(finger_joint_names))
    return torch.cat((arm, fingers), dim=-1), names + list(finger_joint_names)
