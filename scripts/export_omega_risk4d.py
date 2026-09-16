"""Export RGB-only Omega geometry with kinematic metric-scale anchoring."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.robot_flow import PointWorldRobotFlowAdapter, rlbench_franka_joint_trajectory
from scripts.export_pointworld_robot_flow import ARM_JOINTS, FINGER_JOINTS


def transform_points(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat((points, ones), dim=-1)
    return torch.einsum("...ij,...nj->...ni", matrix, homogeneous)[..., :3]


def preprocess_rgb(rgb: np.ndarray, resolution: int, device: torch.device) -> torch.Tensor:
    images = torch.from_numpy(rgb).float().to(device) / 255.0
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("RGB history must have shape [K,3,H,W]")
    return F.interpolate(images, size=(resolution, resolution), mode="bicubic", align_corners=False)


def omega_depth(model, rgb: np.ndarray, resolution: int, device: torch.device):
    images = preprocess_rgb(rgb, resolution, device)
    prediction = model(images)
    depth = prediction["depth"][0]
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    confidence = prediction.get("depth_conf")
    if confidence is None:
        confidence = torch.ones_like(depth)
    else:
        confidence = confidence[0]
        if confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        confidence = confidence.float().sigmoid()
    return depth.float(), confidence.float(), images.shape[-2:]


def scaled_intrinsics(intrinsics: np.ndarray, source_hw, target_hw, device):
    result = torch.from_numpy(intrinsics).float().to(device).clone()
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    result[:, 0] *= target_w / source_w
    result[:, 1] *= target_h / source_h
    return result


def kinematic_scale(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    robot_base_to_world: torch.Tensor,
    robot_points: torch.Tensor,
    min_anchors: int,
) -> tuple[float, int]:
    """Estimate one robust depth scale from projected URDF robot surfaces."""
    ratios = []
    h, w = depth.shape[-2:]
    world_to_camera = torch.linalg.inv(camera_to_world)
    for frame in range(depth.shape[0]):
        world_points = transform_points(robot_points[frame], robot_base_to_world)
        camera_points = transform_points(world_points, world_to_camera[frame])
        z = camera_points[:, 2]
        u = intrinsics[frame, 0, 0] * camera_points[:, 0] / z + intrinsics[frame, 0, 2]
        v = intrinsics[frame, 1, 1] * camera_points[:, 1] / z + intrinsics[frame, 1, 2]
        valid = (z > 0.02) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid].round().long().clamp(0, w - 1)
        v = v[valid].round().long().clamp(0, h - 1)
        z = z[valid]
        if z.numel() == 0:
            continue
        # Keep the nearest sampled robot surface at each pixel to suppress self-occlusion.
        nearest: dict[int, float] = {}
        for pixel, known_depth in zip((v * w + u).tolist(), z.tolist()):
            nearest[pixel] = min(nearest.get(pixel, float("inf")), known_depth)
        pixels = torch.tensor(list(nearest), device=depth.device, dtype=torch.long)
        known = torch.tensor(list(nearest.values()), device=depth.device)
        predicted = depth[frame].flatten()[pixels]
        valid_depth = torch.isfinite(predicted) & (predicted > 1e-4)
        ratio = known[valid_depth] / predicted[valid_depth]
        ratio = ratio[(ratio > 0.05) & (ratio < 20.0)]
        ratios.append(ratio)
    if not ratios:
        raise RuntimeError("kinematic scale anchor found no projected robot pixels")
    ratios = torch.cat(ratios)
    if ratios.numel() < min_anchors:
        raise RuntimeError(
            f"kinematic scale anchor has {ratios.numel()} pixels; need {min_anchors}"
        )
    log_ratio = ratios.log()
    median = log_ratio.median()
    deviation = (log_ratio - median).abs()
    threshold = max(0.15, float(2.5 * deviation.median()))
    inliers = deviation <= threshold
    if inliers.sum() < min_anchors:
        raise RuntimeError("kinematic scale anchor rejected too many pixels")
    return float(log_ratio[inliers].median().exp()), int(inliers.sum())


def depth_to_robot_points(depth, confidence, intrinsics, camera_to_world,
                          robot_base_to_world, count, scale):
    k, h, w = depth.shape
    ids = torch.linspace(0, h * w - 1, count, device=depth.device).long()
    u, v = (ids % w).float(), torch.div(ids, w, rounding_mode="floor").float()
    robot_from_world = torch.linalg.inv(robot_base_to_world)
    result = []
    for frame in range(k):
        z = depth[frame].flatten()[ids] * scale
        camera_points = torch.stack((
            (u - intrinsics[frame, 0, 2]) * z / intrinsics[frame, 0, 0],
            (v - intrinsics[frame, 1, 2]) * z / intrinsics[frame, 1, 1], z,
        ), dim=-1)
        world_points = transform_points(camera_points, camera_to_world[frame])
        result.append(transform_points(world_points, robot_from_world))
    sampled_confidence = confidence.reshape(k, -1)[:, ids]
    return torch.stack(result), sampled_confidence[..., None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default="~/VGGT-Omega")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pointworld-root", default="third_party/PointWorld")
    parser.add_argument("--urdf", default="third_party/RLBench/urdfs/panda/panda.urdf")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--robot-points", type=int, default=128)
    parser.add_argument("--scale-robot-points", type=int, default=2048)
    parser.add_argument("--min-scale-anchors", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source, destination = Path(args.input), Path(args.output)
    files = sorted(source.glob("**/*.npz"))
    if not files:
        raise ValueError(f"no NPZ clips found in {source}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(repo))
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().to(device).eval()
    state = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = PointWorldRobotFlowAdapter(
        args.pointworld_root, args.urdf, args.scale_robot_points, str(device),
        gripper_only=False,
    )

    with torch.inference_mode():
        for file_path in files:
            with np.load(file_path, allow_pickle=False) as archive:
                sample = {key: np.array(archive[key], copy=True) for key in archive.files}
            required = (
                "history_rgb", "joint_history", "candidate_actions", "current_joints",
                "camera_intrinsics", "camera_extrinsics", "robot_base_to_world",
                "future_point_flow",
            )
            missing = [key for key in required if key not in sample]
            if missing:
                raise ValueError(f"{file_path} is missing Omega export inputs: {missing}")
            joint_history = torch.from_numpy(sample["joint_history"]).float()[None, None]
            joint_history, joint_names = rlbench_franka_joint_trajectory(
                joint_history, ARM_JOINTS, FINGER_JOINTS,
            )
            surface = adapter.surface_points(joint_history[0, 0].to(device), joint_names)
            base_to_world = torch.from_numpy(sample["robot_base_to_world"]).float().to(device)

            def export_sequence(rgb, camera_extrinsics):
                depth, confidence, output_hw = omega_depth(model, rgb, args.resolution, device)
                intrinsics = scaled_intrinsics(
                    sample["camera_intrinsics"], rgb.shape[-2:], output_hw, device,
                )
                camera_to_world = torch.from_numpy(camera_extrinsics).float().to(device)
                scale, anchor_count = kinematic_scale(
                    depth, intrinsics, camera_to_world, base_to_world, surface,
                    args.min_scale_anchors,
                )
                points, features = depth_to_robot_points(
                    depth, confidence, intrinsics, camera_to_world, base_to_world,
                    sample["future_point_flow"].shape[2], scale,
                )
                return points.cpu().numpy(), features.cpu().numpy(), scale, anchor_count

            points, point_features, scale, anchors = export_sequence(
                sample["history_rgb"], sample["camera_extrinsics"])
            sample["points"] = points.astype(np.float32)
            sample["point_features"] = point_features.astype(np.float32)
            sample["omega_scale"] = np.asarray(scale, np.float32)
            sample["omega_scale_anchor_count"] = np.asarray(anchors, np.int32)
            if "view_rgb_history" in sample:
                view_points, view_features, view_scale, view_anchors = [], [], [], []
                for view_index in range(len(sample["view_rgb_history"])):
                    exported = export_sequence(
                        sample["view_rgb_history"][view_index],
                        sample["view_camera_extrinsics"][view_index],
                    )
                    view_points.append(exported[0]); view_features.append(exported[1])
                    view_scale.append(exported[2]); view_anchors.append(exported[3])
                sample["view_points"] = np.stack(view_points).astype(np.float32)
                sample["view_point_features"] = np.stack(view_features).astype(np.float32)
                sample["view_omega_scale"] = np.asarray(view_scale, np.float32)
                sample["view_omega_scale_anchor_count"] = np.asarray(view_anchors, np.int32)

            candidate = torch.from_numpy(sample["candidate_actions"]).float()[None]
            candidate_joints, candidate_names = rlbench_franka_joint_trajectory(
                candidate, ARM_JOINTS, FINGER_JOINTS,
            )
            current = torch.from_numpy(sample["current_joints"]).float()[None, None, None]
            current_joints, _ = rlbench_franka_joint_trajectory(
                current, ARM_JOINTS, FINGER_JOINTS,
            )
            full_robot_flow = adapter.encode(
                candidate_joints, current_joints[:, 0, 0], candidate_names,
            )[0]
            ids = torch.linspace(
                0, full_robot_flow.shape[-2] - 1, args.robot_points,
                device=full_robot_flow.device,
            ).long()
            sample["robot_action_flow"] = full_robot_flow[..., ids, :].cpu().numpy().astype(np.float32)

            world_to_robot = np.linalg.inv(sample["robot_base_to_world"])
            sample["future_point_flow"] = (
                sample["future_point_flow"] @ world_to_robot[:3, :3].T
            ).astype(np.float32)
            sample["coordinate_frame"] = np.asarray("robot_base")
            sample["geometry_source"] = np.asarray("VGGT-Omega-kinematic-scale")
            output_file = destination / file_path.relative_to(source)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, **sample)
            print(f"wrote {output_file} omega_scale={scale:.5f} anchors={anchors}")


if __name__ == "__main__":
    main()
