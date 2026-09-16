"""Verify PointWorld URDF sampling and robot-flow shape without a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.robot_flow import PointWorldRobotFlowAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointworld-root", default="third_party/PointWorld")
    parser.add_argument("--urdf", default="third_party/RLBench/urdfs/panda/panda.urdf")
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    adapter = PointWorldRobotFlowAdapter(
        args.pointworld_root, args.urdf, args.points, args.device,
    )
    current = torch.zeros(1, len(adapter.joint_names))
    trajectory = torch.zeros(1, 2, 3, len(adapter.joint_names))
    flow = adapter.encode(trajectory, current)
    expected = (1, 2, 3, args.points, 3)
    if tuple(flow.shape) != expected or float(flow.abs().max()) > 1e-6:
        raise RuntimeError(f"unexpected robot flow: shape={tuple(flow.shape)} max={flow.abs().max()}")
    moving = trajectory.clone()
    moving[..., 0] = 0.1
    moving_flow = adapter.encode(moving, current)
    if float(moving_flow.abs().max()) <= 1e-6:
        raise RuntimeError("nonzero joint motion produced zero robot flow")
    print(f"joints={adapter.joint_names}")
    print(f"robot_flow_shape={tuple(flow.shape)} zero_and_nonzero_flow_ok=true")


if __name__ == "__main__":
    main()
