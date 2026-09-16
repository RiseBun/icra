"""Compute task-stratified statistics for a public trajectory audit.

The input audit already contains episode-level window rows and labels.  This
script never resamples windows independently: confidence intervals resample
whole episodes within each task stratum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _episode_bootstrap(y, score, groups, n=1000, seed=2027):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    values = []
    for _ in range(int(n)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        if len(np.unique(y[idx])) > 1:
            values.append(roc_auc_score(y[idx], score[idx]))
    if not values:
        return None
    return [float(x) for x in np.percentile(values, [2.5, 50.0, 97.5])]


def _summarize(rows, bootstrap, seed):
    if not rows:
        return {"status": "empty", "n_windows": 0, "n_episodes": 0}
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    amp = np.asarray([float(r["mean_amp"]) for r in rows], dtype=float)
    episodes = np.asarray([str(r["episode_id"]) for r in rows])
    out = {
        "status": "ok" if len(np.unique(y)) > 1 else "no_label_variation",
        "n_windows": int(len(rows)),
        "n_episodes": int(len(np.unique(episodes))),
        "positive_rate": float(y.mean()),
    }
    if len(np.unique(y)) > 1:
        out["amplitude_auroc"] = float(roc_auc_score(y, amp))
        out["amplitude_auroc_ci95"] = _episode_bootstrap(
            y, amp, episodes, n=bootstrap, seed=seed
        )
    else:
        out["amplitude_auroc"] = None
        out["amplitude_auroc_ci95"] = None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-key", default="skill")
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()

    data = json.loads(Path(args.audit).read_text())
    rows = [r for r in data.get("rows", []) if r.get("label") is not None]
    grouped = {}
    for row in rows:
        key = str(row.get(args.group_key, "unknown"))
        grouped.setdefault(key, []).append(row)

    summaries = {}
    for index, (key, group_rows) in enumerate(sorted(grouped.items())):
        summaries[key] = _summarize(group_rows, args.bootstrap, 2027 + index)

    valid = [v for v in summaries.values() if v.get("amplitude_auroc") is not None]
    above = [v for v in valid if v["amplitude_auroc"] > 0.5]
    below = [v for v in valid if v["amplitude_auroc"] < 0.5]
    output = {
        "status": "ok",
        "audit": str(args.audit),
        "group_key": args.group_key,
        "n_groups": int(len(summaries)),
        "n_groups_with_label_variation": int(len(valid)),
        "n_groups_above_chance": int(len(above)),
        "n_groups_below_chance": int(len(below)),
        "groups": summaries,
        "limitations": [
            "Confidence intervals resample complete episodes, never independent windows.",
            "This conversion exposes task metadata but no validated environment identifier.",
            "Progress-proxy labels are not verified success/failure labels.",
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
