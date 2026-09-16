"""Gate RGB-only Omega geometry against evaluation-only simulator geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def transform(points, matrix):
    ones = np.ones((*points.shape[:-1], 1), np.float32)
    return (np.concatenate((points, ones), axis=-1) @ matrix.T)[..., :3]


def evaluate(raw_root: Path, exported_root: Path, max_mean_epe: float = 0.15,
             max_median_epe: float = 0.10, max_p90_epe: float = 0.30,
             min_anchor_count: int = 16) -> dict:
    raw_files = sorted(raw_root.glob("**/*.npz"))
    if not raw_files:
        raise ValueError(f"no raw clips found in {raw_root}")
    all_errors, scales, anchors, clips, resolutions = [], [], [], [], set()
    for raw_path in raw_files:
        exported_path = exported_root / raw_path.relative_to(raw_root)
        if not exported_path.is_file():
            raise FileNotFoundError(exported_path)
        with np.load(raw_path, allow_pickle=False) as raw, np.load(
                exported_path, allow_pickle=False) as exported:
            world_to_robot = np.linalg.inv(raw["robot_base_to_world"])
            target = transform(raw["points"], world_to_robot)
            prediction = exported["points"]
            if prediction.shape != target.shape:
                raise ValueError(f"point shape mismatch in {raw_path}")
            error = np.linalg.norm(prediction - target, axis=-1).reshape(-1)
            scale = float(exported["omega_scale"])
            anchor_count = int(exported["omega_scale_anchor_count"])
            resolution = [int(value) for value in raw["history_rgb"].shape[-2:]]
            resolutions.add(tuple(resolution))
            clips.append({
                "path": str(raw_path.relative_to(raw_root)),
                "source_resolution": resolution,
                "point_epe_mean_m": float(error.mean()),
                "point_epe_median_m": float(np.median(error)),
                "point_epe_p90_m": float(np.quantile(error, 0.9)),
                "omega_scale": scale,
                "scale_anchor_count": anchor_count,
            })
            all_errors.append(error)
            scales.append(scale)
            anchors.append(anchor_count)
    error = np.concatenate(all_errors)
    metrics = {
        "point_epe_mean_m": float(error.mean()),
        "point_epe_median_m": float(np.median(error)),
        "point_epe_p90_m": float(np.quantile(error, 0.9)),
        "omega_scale_mean": float(np.mean(scales)),
        "omega_scale_std": float(np.std(scales)),
        "scale_anchor_count_min": int(np.min(anchors)),
        "scale_anchor_count_mean": float(np.mean(anchors)),
    }
    thresholds = {
        "max_mean_epe_m": max_mean_epe,
        "max_median_epe_m": max_median_epe,
        "max_p90_epe_m": max_p90_epe,
        "min_scale_anchor_count": min_anchor_count,
    }
    checks = {
        "mean_epe": metrics["point_epe_mean_m"] <= max_mean_epe,
        "median_epe": metrics["point_epe_median_m"] <= max_median_epe,
        "p90_epe": metrics["point_epe_p90_m"] <= max_p90_epe,
        "scale_anchors": metrics["scale_anchor_count_min"] >= min_anchor_count,
        "source_is_256": resolutions == {(256, 256)},
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "clips": len(clips),
        "source_resolutions": [list(value) for value in sorted(resolutions)],
        "metrics": metrics,
        "per_clip": clips,
        "note": "simulator geometry is evaluation-only and was not used by the exporter",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--exported", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-mean-epe", type=float, default=0.15)
    parser.add_argument("--max-median-epe", type=float, default=0.10)
    parser.add_argument("--max-p90-epe", type=float, default=0.30)
    parser.add_argument("--min-anchor-count", type=int, default=16)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        Path(args.raw), Path(args.exported), args.max_mean_epe,
        args.max_median_epe, args.max_p90_epe, args.min_anchor_count)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if args.enforce and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
