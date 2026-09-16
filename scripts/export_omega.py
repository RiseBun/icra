"""Export VGGT-Omega predictions to the ICRA 2027 feature contract.

This adapter intentionally stores camera-frame points first. A robot calibration
file can later transform them into robot_base without changing the model API.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch


def _sample_points(depth: torch.Tensor, intrinsics: torch.Tensor, count: int):
    # depth: [K,H,W], intrinsics: [K,3,3]
    k, h, w = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device),
        torch.arange(w, device=depth.device), indexing="ij"
    )
    flat = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    idx = torch.linspace(0, flat.shape[0] - 1, count, device=depth.device).long()
    pix = flat[idx]
    u, v = pix[:, 0].float(), pix[:, 1].float()
    out = []
    for i in range(k):
        z = depth[i].flatten()[idx].clamp_min(1e-4)
        fx, fy = intrinsics[i, 0, 0], intrinsics[i, 1, 1]
        cx, cy = intrinsics[i, 0, 2], intrinsics[i, 1, 2]
        out.append(torch.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), dim=-1))
    return torch.stack(out, dim=0)


def _load_rigid_transform(path: str) -> torch.Tensor:
    p = Path(path).expanduser()
    if p.suffix.lower() == ".json":
        value = json.loads(p.read_text())
    else:
        value = np.load(p)
    matrix = torch.as_tensor(value, dtype=torch.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera-to-robot transform must be [4,4], got {tuple(matrix.shape)}")
    return matrix


def _apply_rigid_transform(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat((points, ones), dim=-1)
    return torch.einsum("ij,knj->kni", matrix.to(points.device), homogeneous)[..., :3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="~/VGGT-Omega")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument(
        "--camera-to-robot",
        default=None,
        help="optional 4x4 camera-to-robot transform (.npy or .json)",
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    import sys
    sys.path.insert(0, str(repo))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    paths = sorted(glob.glob(str(Path(args.input).expanduser() / "*.jpg")))
    paths += sorted(glob.glob(str(Path(args.input).expanduser() / "*.png")))
    if not paths:
        raise SystemExit("No .jpg/.png frames found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGTOmega().to(device).eval()
    state = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    images = load_and_preprocess_images(paths, image_resolution=args.resolution).to(device)
    with torch.inference_mode():
        pred = model(images)
        extrinsics, intrinsics = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    depth = pred["depth"][0, ..., 0]
    conf = pred["depth_conf"][0]
    points = _sample_points(depth, intrinsics[0], args.points)
    coordinate_frame = "camera"
    if args.camera_to_robot:
        points = _apply_rigid_transform(points, _load_rigid_transform(args.camera_to_robot))
        coordinate_frame = "robot_base"
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points=points.float().cpu().numpy(),
        depth=depth.float().cpu().numpy(),
        depth_conf=conf.float().cpu().numpy(),
        intrinsics=intrinsics[0].float().cpu().numpy(),
        extrinsics=extrinsics[0].float().cpu().numpy(),
        frame_paths=np.asarray(paths),
        coordinate_frame=np.asarray(coordinate_frame),
        backbone=np.asarray("VGGT-Omega-1B-512"),
    )
    print(f"saved {out} points={tuple(points.shape)} depth={tuple(depth.shape)}")


if __name__ == "__main__":
    main()
