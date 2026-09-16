"""Check RLBench camera projection conventions independently of Omega quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def transform(points, matrix):
    ones = np.ones((*points.shape[:-1], 1), np.float32)
    return (np.concatenate((points, ones), axis=-1) @ matrix.T)[..., :3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--exported")
    args = parser.parse_args()
    with np.load(Path(args.raw), allow_pickle=False) as data:
        raw = {key: np.asarray(data[key]) for key in data.files}
    points = raw["points"]
    k, n = points.shape[:2]
    h, w = raw["history_rgb"].shape[-2:]
    ids = np.linspace(0, h * w - 1, n).astype(np.int64)
    expected_u, expected_v = ids % w, ids // w
    pixel_error, reconstruction_error, z_values = [], [], []
    for frame in range(k):
        camera = transform(points[frame], np.linalg.inv(raw["camera_extrinsics"][frame]))
        intrinsics = raw["camera_intrinsics"][frame]
        u = intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]
        v = intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]
        pixel_error.append(np.sqrt((u - expected_u) ** 2 + (v - expected_v) ** 2))
        reconstructed_camera = np.stack((
            (expected_u - intrinsics[0, 2]) * camera[:, 2] / intrinsics[0, 0],
            (expected_v - intrinsics[1, 2]) * camera[:, 2] / intrinsics[1, 1],
            camera[:, 2],
        ), -1)
        reconstructed_world = transform(reconstructed_camera, raw["camera_extrinsics"][frame])
        reconstruction_error.append(np.linalg.norm(reconstructed_world - points[frame], axis=-1))
        z_values.append(camera[:, 2])
    result = {
        "projection_pixel_error_mean": float(np.mean(pixel_error)),
        "projection_pixel_error_p90": float(np.quantile(pixel_error, 0.9)),
        "roundtrip_epe_mean_m": float(np.mean(reconstruction_error)),
        "camera_z_median_m": float(np.median(z_values)),
        "camera_z_min_max_m": [float(np.min(z_values)), float(np.max(z_values))],
        "intrinsics_first": raw["camera_intrinsics"][0].tolist(),
    }
    if args.exported:
        with np.load(args.exported, allow_pickle=False) as exported:
            prediction_world = transform(exported["points"], raw["robot_base_to_world"])
        predicted_camera = transform(
            prediction_world[-1], np.linalg.inv(raw["camera_extrinsics"][-1]))
        result["omega_camera_z_median_m"] = float(np.median(predicted_camera[:, 2]))
        result["omega_camera_z_min_max_m"] = [
            float(predicted_camera[:, 2].min()), float(predicted_camera[:, 2].max())]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
