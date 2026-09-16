"""Audit whether counterfactual candidates contain useful action variation.

The audit is intentionally model-free.  It catches a common failure mode in
counterfactual collection: all candidates are nearly identical, so an action
conditioned world model can minimize loss with a passive prior and cannot be
evaluated fairly.  The script accepts raw collection files as well as later
PointWorld/Omega caches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _pairwise(values: np.ndarray) -> np.ndarray:
    """Return upper-triangular pairwise mean L2 distances over candidates."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError(f"candidate array must have at least 2 dims, got {values.shape}")
    candidates = values.shape[0]
    flat = values.reshape(candidates, -1)
    distances = []
    for i in range(candidates):
        for j in range(i + 1, candidates):
            distances.append(float(np.linalg.norm(flat[i] - flat[j]) / np.sqrt(flat.shape[1])))
    return np.asarray(distances, dtype=np.float32)


def _stats(values: np.ndarray | None) -> dict[str, float | None]:
    if values is None or values.size == 0:
        return {"mean": None, "median": None, "max": None, "p90": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _label_summary(archive: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("success", "collision", "failure", "task_success", "harmful_collision"):
        if key not in archive:
            continue
        values = np.asarray(archive[key]).reshape(-1)
        unique, counts = np.unique(values, return_counts=True)
        result[key] = {
            "unique": [float(v) if np.issubdtype(unique.dtype, np.number) else str(v) for v in unique],
            "counts": [int(v) for v in counts],
            "varies": bool(unique.size > 1),
        }
    return result


def audit_file(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        action_key = "candidate_actions" if "candidate_actions" in keys else None
        flow_key = "robot_action_flow" if "robot_action_flow" in keys else None
        future_flow_key = "future_point_flow" if "future_point_flow" in keys else None
        future_latent_key = "omega_future_latent_tokens" if "omega_future_latent_tokens" in keys else None

        action_dist = _pairwise(archive[action_key]) if action_key else None
        robot_dist = _pairwise(archive[flow_key]) if flow_key else None
        future_dist = _pairwise(archive[future_flow_key]) if future_flow_key else None
        latent_dist = _pairwise(archive[future_latent_key]) if future_latent_key else None
        labels = _label_summary(archive)
        informative = any(item.get("varies", False) for item in labels.values())
        return {
            "file": str(path),
            "candidates": int(archive[action_key].shape[0]) if action_key else int(
                archive[flow_key].shape[0]) if flow_key else int(
                archive[future_flow_key].shape[0]) if future_flow_key else None,
            "action_pairwise": _stats(action_dist),
            "robot_flow_pairwise": _stats(robot_dist),
            "future_flow_pairwise": _stats(future_dist),
            "future_latent_pairwise": _stats(latent_dist),
            "labels": labels,
            "has_label_variation": informative,
        }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(name: str) -> np.ndarray:
        return np.asarray([
            item[name]["mean"] for item in records
            if item[name]["mean"] is not None
        ], dtype=np.float32)

    summaries = {}
    for name in ("action_pairwise", "robot_flow_pairwise", "future_flow_pairwise", "future_latent_pairwise"):
        summaries[name] = _stats(collect(name))
    return {
        "files": len(records),
        "informative_label_files": int(sum(item["has_label_variation"] for item in records)),
        "aggregated_decision_means": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    files = sorted(Path(args.data).glob("**/*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"no NPZ files found in {args.data}")
    records = [audit_file(path) for path in files]
    result = {"data": str(args.data), "summary": _aggregate(records), "files": records}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


if __name__ == "__main__":
    main()
