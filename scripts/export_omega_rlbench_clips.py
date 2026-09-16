"""Export Omega-aligned point clips with RLBench future-flow labels."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch


def transform(points, matrix):
    ones = np.ones((*points.shape[:-1], 1), dtype=np.float32)
    return (np.concatenate((points, ones), -1) @ matrix.T)[..., :3]


def sample_grid(cloud, count):
    t, h, w, _ = cloud.shape
    ids = np.linspace(0, h * w - 1, count).astype(np.int64)
    return cloud.reshape(t, h * w, 3)[:, ids]


def sample_at_uv(cloud, uv, target_hw):
    """Sample a native-resolution cloud at pixels from the Omega output grid."""
    t, h, w, _ = cloud.shape
    th, tw = target_hw
    u = np.clip(np.rint(uv[:, 0] * w / tw), 0, w - 1).astype(np.int64)
    v = np.clip(np.rint(uv[:, 1] * h / th), 0, h - 1).astype(np.int64)
    return cloud[:, v, u]


def umeyama_similarity(source, target):
    """Return a 4x4 similarity map source -> target."""
    src_mu, tgt_mu = source.mean(0), target.mean(0)
    src0, tgt0 = source - src_mu, target - tgt_mu
    cov = (tgt0.T @ src0) / max(1, len(source))
    u, _, vh = np.linalg.svd(cov)
    s = np.eye(3, dtype=np.float32)
    if np.linalg.det(u @ vh) < 0:
        s[-1, -1] = -1.0
    r = (u @ s @ vh).astype(np.float32)
    var = (src0 ** 2).sum() / max(1, len(source))
    scale = float(np.trace(np.diag(np.linalg.svd(cov, compute_uv=False)) @ s) / max(var, 1e-8))
    t = tgt_mu - scale * (r @ src_mu)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = scale * r
    matrix[:3, 3] = t
    return matrix, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory")
    ap.add_argument("--repo", default="~/VGGT-Omega")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--future", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()
    traj = Path(args.trajectory).expanduser()
    data = np.load(traj)
    image_paths = sorted(glob.glob(str(traj.parent / "frame_*.png")))
    if not image_paths:
        raise SystemExit(f"no RGB frames next to {traj}")
    if len(image_paths) != len(data["front_point_cloud"]):
        raise SystemExit("RGB/point-cloud frame count mismatch")
    repo = Path(args.repo).expanduser().resolve()
    import sys
    sys.path.insert(0, str(repo))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGTOmega().to(device).eval()
    state = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    gt_cloud = np.asarray(data["front_point_cloud"], dtype=np.float32)
    extrinsics_key = "front_camera_extrinsics" if "front_camera_extrinsics" in data.files else "camera_extrinsics"
    gt_cam_to_world = np.asarray(data[extrinsics_key], dtype=np.float32)
    gt_world_to_ref = np.linalg.inv(gt_cam_to_world)
    gt_ref_full = transform(gt_cloud, gt_world_to_ref)
    actions = np.asarray(data["actions"], dtype=np.float32)
    low_dim = np.asarray(data["low_dim"], dtype=np.float32)
    success = np.asarray(data.get("success", np.zeros(len(gt_cloud))), dtype=np.float32)
    collision = np.asarray(data.get("collision", np.zeros(len(gt_cloud))), dtype=np.float32)
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    counter = 0
    with torch.inference_mode():
        for start in range(0, len(image_paths) - args.history - args.future + 1, args.stride):
            paths = image_paths[start:start + args.history]
            images = load_and_preprocess_images(paths, image_resolution=args.resolution).to(device)
            pred = model(images)
            extrinsics, intrinsics = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
            depth = pred["depth"][0, ..., 0]
            depth_conf = pred.get("depth_conf")
            if depth_conf is not None:
                depth_conf = depth_conf[0]
            k, h, w = depth.shape
            yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            flat = torch.stack((xx.flatten(), yy.flatten()), -1)
            ids = torch.linspace(0, flat.shape[0] - 1, args.points, device=device).long()
            uv = flat[ids].float()
            pred_points = []
            pred_conf = []
            for frame in range(k):
                z = depth[frame].flatten()[ids].clamp_min(1e-4)
                fx, fy = intrinsics[0, frame, 0, 0], intrinsics[0, frame, 1, 1]
                cx, cy = intrinsics[0, frame, 0, 2], intrinsics[0, frame, 1, 2]
                pred_points.append(torch.stack(((uv[:, 0] - cx) * z / fx, (uv[:, 1] - cy) * z / fy, z), -1))
                if depth_conf is not None:
                    pred_conf.append(depth_conf[frame].flatten()[ids])
            pred_points = torch.stack(pred_points).cpu().numpy()
            point_confidence = (
                torch.stack(pred_conf).float().sigmoid().cpu().numpy()
                if pred_conf else np.ones((k, args.points), dtype=np.float32)
            )
            pred_extr = extrinsics[0].cpu().numpy()
            pred_extr_h = np.concatenate(
                (pred_extr, np.broadcast_to(np.array([0, 0, 0, 1], np.float32), (k, 1, 4))), axis=1
            )
            ref_extr = pred_extr_h[0]
            pred_ref = np.stack([transform(pred_points[i], ref_extr @ np.linalg.inv(pred_extr_h[i])) for i in range(k)])
            gt_uv = uv.detach().cpu().numpy()
            gt_ref = sample_at_uv(gt_ref_full, gt_uv, (h, w))
            calibration, scale = umeyama_similarity(pred_ref[0], gt_ref[start])
            pred_ref = transform(pred_ref, calibration)
            hs = start + args.history
            fs = hs + args.future
            future = gt_ref[hs:fs][None]
            # Convert future simulator points into the first camera reference frame.
            future = future
            anchor = pred_ref[-1]
            disp = np.linalg.norm(future[0] - gt_ref[hs - 1], axis=-1).mean(0)
            affordance = (disp > np.quantile(disp, 0.75)).astype(np.float32)[None]
            np.savez_compressed(
                out / f"clip_{counter:06d}.npz", points=pred_ref,
                point_confidence=point_confidence,
                robot_state=low_dim[hs - 1], actions=actions[hs - 1:hs],
                # actions[t] is the command applied after observation t in the
                # replay trajectory saved by the collector.  From state
                # hs-1, the first future command is therefore hs-1.
                action_chunk=actions[hs - 1:fs - 1],
                future_points=future, affordance=affordance,
                success=np.asarray([float(success.max() > 0.5)], np.float32),
                collision=np.asarray([float(collision[hs:fs].max() > 0.5)], np.float32),
                coordinate_frame=np.asarray("omega_ref_calibrated"), calibration_scale=np.asarray(scale, np.float32), source=np.asarray(str(traj)),
            )
            counter += 1
    print(f"wrote {counter} Omega clips to {out}")


if __name__ == "__main__":
    main()
