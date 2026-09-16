"""Offline PointWorld FK export from RLBench candidate actions to v2 robot flow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.robot_flow import PointWorldRobotFlowAdapter, rlbench_franka_joint_trajectory


ARM_JOINTS = [f"Pandajoint{index}" for index in range(1, 8)]
FINGER_JOINTS = ["Pandagripperjoint1", "Pandagripperjoint2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ file or directory")
    parser.add_argument("--output", required=True, help="output file or directory")
    parser.add_argument("--pointworld-root", default="third_party/PointWorld")
    parser.add_argument("--urdf", default="third_party/RLBench/urdfs/panda/panda.urdf")
    parser.add_argument("--action-key", default="candidate_actions")
    parser.add_argument("--current-joints-key", default="current_joints")
    parser.add_argument("--action-mode", choices=("delta", "absolute"), default="absolute")
    parser.add_argument("--robot-points", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm-only", action="store_true")
    parser.add_argument("--gripper-open-position", type=float, default=0.04)
    args = parser.parse_args()
    source = Path(args.input)
    files = [source] if source.is_file() else sorted(source.glob("**/*.npz"))
    if not files:
        raise ValueError(f"no NPZ files found in {source}")
    destination = Path(args.output)
    adapter = PointWorldRobotFlowAdapter(
        args.pointworld_root, args.urdf, args.robot_points, args.device,
        gripper_only=False,
    )

    for file_path in files:
        with np.load(file_path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        if args.action_key not in arrays or args.current_joints_key not in arrays:
            raise ValueError(
                f"{file_path} must contain {args.action_key} and explicit "
                f"{args.current_joints_key}; low-dimensional state layout is not inferred"
            )
        actions = torch.from_numpy(arrays[args.action_key]).float()
        current = torch.from_numpy(arrays[args.current_joints_key]).float()
        if actions.ndim != 3 or current.ndim != 1 or current.numel() < 7:
            raise ValueError("expected actions [M,L,8] and current_joints [8]")
        actions = actions.unsqueeze(0)
        if args.action_mode == "delta":
            arm_absolute = current[:7][None, None, None] + actions[..., :7].cumsum(dim=2)
            actions = torch.cat((arm_absolute, actions[..., 7:8]), dim=-1)
        finger_names = [] if args.arm_only else FINGER_JOINTS
        joint_trajectory, joint_names = rlbench_franka_joint_trajectory(
            actions, ARM_JOINTS, finger_names, args.gripper_open_position,
        )
        current_action = current[:8][None, None, None]
        current_trajectory, _ = rlbench_franka_joint_trajectory(
            current_action, ARM_JOINTS, finger_names, args.gripper_open_position,
        )
        robot_flow = adapter.encode(
            joint_trajectory, current_trajectory[:, 0, 0], joint_names,
        )[0].detach().cpu().numpy().astype(np.float32)
        arrays["robot_action_flow"] = robot_flow
        if source.is_file():
            output_file = destination
        else:
            output_file = destination / file_path.relative_to(source)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, **arrays)
        print(f"wrote {output_file}")


if __name__ == "__main__":
    main()
