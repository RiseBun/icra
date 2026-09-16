"""Audit v2 label quality before any risk training or paper evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import validate_risk4d_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--min-risk-decisions", type=int, default=1)
    parser.add_argument("--require-risk-variation", action="store_true")
    parser.add_argument("--paper-mode", action="store_true")
    args = parser.parse_args()
    files = sorted(Path(args.data).glob("**/*.npz"))
    if not files:
        raise ValueError(f"no NPZ files found in {args.data}")
    decisions_with_success_variation = 0
    decisions_with_collision_variation = 0
    flow_magnitude, contact_positive, harmful_positive = [], [], []
    candidates = []
    view_counts, view_geometry_diversity = [], []
    candidate_sources, geometry_sources, coordinate_frames = set(), set(), set()
    for file_path in files:
        with np.load(file_path, allow_pickle=False) as archive:
            sample = {key: np.asarray(archive[key]) for key in archive.files}
        validate_risk4d_sample(sample)
        candidates.append(len(sample["success"]))
        decisions_with_success_variation += int(np.unique(sample["success"]).size > 1)
        decisions_with_collision_variation += int(np.unique(sample["collision"]).size > 1)
        flow_magnitude.append(np.linalg.norm(sample["future_point_flow"], axis=-1).mean())
        contact_positive.append(sample["task_contact_map"].mean())
        harmful_positive.append(sample["harmful_collision_map"].mean())
        if "view_points" in sample:
            view_counts.append(len(sample["view_points"]))
            reference = sample["view_points"][0]
            if len(sample["view_points"]) > 1:
                view_geometry_diversity.append(float(np.mean([
                    np.linalg.norm(view - reference, axis=-1).mean()
                    for view in sample["view_points"][1:]
                ])))
        candidate_sources.add(str(sample.get("candidate_source", "missing")))
        geometry_sources.add(str(sample.get("geometry_source", "simulator_or_missing")))
        coordinate_frames.add(str(sample.get("coordinate_frame", "missing")))
    risk_decisions = sum(
        1 for file_path in files
        if _has_risk_variation(file_path)
    )
    result = {
        "clips": len(files),
        "candidate_count_min_max": [int(min(candidates)), int(max(candidates))],
        "decisions_with_success_variation": decisions_with_success_variation,
        "decisions_with_collision_variation": decisions_with_collision_variation,
        "decisions_with_any_risk_variation": risk_decisions,
        "mean_flow_magnitude": float(np.mean(flow_magnitude)),
        "task_contact_positive_fraction": float(np.mean(contact_positive)),
        "harmful_collision_positive_fraction": float(np.mean(harmful_positive)),
        "usable_for_geometry": bool(np.mean(flow_magnitude) > 1e-5),
        "usable_for_risk": risk_decisions >= args.min_risk_decisions,
        "view_count_min_max": [int(min(view_counts)), int(max(view_counts))] if view_counts else None,
        "mean_view_geometry_diversity": float(np.mean(view_geometry_diversity))
        if view_geometry_diversity else None,
        "candidate_sources": sorted(candidate_sources),
        "geometry_sources": sorted(geometry_sources),
        "coordinate_frames": sorted(coordinate_frames),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_risk_variation and not result["usable_for_risk"]:
        raise SystemExit(2)
    if args.paper_mode:
        invalid_candidate = any(
            source == "missing" or "debug" in source for source in candidate_sources)
        valid_geometry = all(
            source.startswith("VGGT-Omega") or source.startswith("VGGT4D")
            for source in geometry_sources)
        valid_frame = coordinate_frames == {"robot_base"}
        if invalid_candidate or not valid_geometry or not valid_frame or not result["usable_for_risk"]:
            raise SystemExit(2)


def _has_risk_variation(file_path: Path) -> bool:
    with np.load(file_path, allow_pickle=False) as archive:
        return (np.unique(archive["success"]).size > 1
                or np.unique(archive["collision"]).size > 1)


if __name__ == "__main__":
    main()
