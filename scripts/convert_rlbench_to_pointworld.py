"""Convert risk4d NPZ clips to a PointWorld-compatible tensor contract.

The converter is deliberately lossless for labels: it copies every source key
and adds aliases used by PointWorld.  It does not fabricate robot points or
contact labels, because those would invalidate a paper comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def convert_sample(sample: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("points", "point_features", "robot_action_flow", "future_point_flow")
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"sample is missing required PointWorld inputs: {missing}")
    points = np.asarray(sample["points"], dtype=np.float32)
    features = np.asarray(sample["point_features"], dtype=np.float32)
    robot_flow = np.asarray(sample["robot_action_flow"], dtype=np.float32)
    future = np.asarray(sample["future_point_flow"], dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [K,N,3]")
    if features.shape[:2] != points.shape[:2]:
        raise ValueError("point_features must align with points")
    if robot_flow.ndim != 4 or robot_flow.shape[-1] != 3:
        raise ValueError("robot_action_flow must have shape [M,L,Q,3]")
    if future.ndim != 4 or future.shape[0] != robot_flow.shape[0] or future.shape[2] != points.shape[1]:
        raise ValueError("future_point_flow must have shape [M,H,N,3]")
    output = {key: np.array(value, copy=True) for key, value in sample.items()}
    output.update({
        "scene_points": points,
        "scene_features": features,
        "scene_exists": np.isfinite(points).all(axis=-1).astype(np.float32),
        "robot_flow": robot_flow,
        "robot_exists": np.isfinite(robot_flow).all(axis=-1).all(axis=-1).astype(np.float32),
        "future_scene_flow": future,
        "pointworld_contract_version": np.asarray("pointworld-v1"),
    })
    if "robot_surface_points" in sample:
        output["robot_points"] = np.asarray(sample["robot_surface_points"], dtype=np.float32)
    elif "robot_points" in sample:
        output["robot_points"] = np.asarray(sample["robot_points"], dtype=np.float32)
    else:
        # The official model can require absolute robot points.  Make this
        # explicit instead of silently treating displacement as position.
        output["robot_points"] = np.zeros(robot_flow.shape[::2] + (3,), dtype=np.float32)
        output["robot_points_available"] = np.asarray(False)
    return output


def convert_directory(source: str | Path, destination: str | Path) -> dict:
    source, destination = Path(source), Path(destination)
    files = sorted(source.glob("**/*.npz"))
    if not files:
        raise ValueError(f"no NPZ files found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    names, total = [], 0
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            converted = convert_sample({key: archive[key] for key in archive.files})
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **converted)
        names.append(str(target)); total += 1
    manifest = {
        "source": str(source), "destination": str(destination), "clips": total,
        "files": names, "contract": "pointworld-v1", "labels_copied": True,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(convert_directory(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
