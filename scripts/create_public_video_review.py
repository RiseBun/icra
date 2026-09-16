"""Create a blind, task-stratified video review manifest.

The manifest deliberately separates frames for human review from the proxy
labels used by the audit.  It is a validation aid, not a new ground-truth
label source.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.audit).read_text())
    rows = data.get("rows", [])
    by_episode = {}
    for row in rows:
        match = re.search(r"episode_(\d+)$", str(row["episode_id"]))
        if match:
            by_episode.setdefault(int(match.group(1)), []).append(row)
    episodes = pd.read_parquet(
        args.episodes,
        columns=[
            "episode_index",
            "videos/observation.images.image_0/from_timestamp",
            "videos/observation.images.image_0/to_timestamp",
        ],
    ).set_index("episode_index")
    rng = np.random.default_rng(args.seed)
    candidates = {}
    for ep, erows in by_episode.items():
        if ep not in episodes.index:
            continue
        row = erows[len(erows) // 2]
        task = str(row.get("skill", "unknown"))
        start = float(episodes.loc[ep, "videos/observation.images.image_0/from_timestamp"])
        timestamp = start + float(row.get("window_start", 0)) * 0.2 + 0.7
        candidates.setdefault(task, []).append(
            {
                "review_id": None,
                "episode_index": ep,
                "task": task,
                "timestamp": round(timestamp, 3),
                "window_start": int(row.get("window_start", 0)),
                "frame_source": "observation.images.image_0",
            }
        )
    blind = []
    revealed = []
    index = 0
    for task in sorted(candidates):
        values = candidates[task]
        if len(values) > args.per_task:
            values = [values[i] for i in rng.choice(len(values), args.per_task, replace=False)]
        for item in values:
            item = dict(item)
            item["review_id"] = f"review_{index:04d}"
            blind.append(item)
            matching = by_episode.get(item["episode_index"], [])
            proxy = next(
                (
                    row
                    for row in matching
                    if int(row.get("window_start", -1)) == item["window_start"]
                ),
                {},
            )
            revealed.append(
                {
                    "review_id": item["review_id"],
                    "proxy_label": proxy.get("label"),
                    "progress": proxy.get("progress"),
                    "label_source": proxy.get("label_source"),
                }
            )
            index += 1
    if any(item["proxy_label"] is None for item in revealed):
        raise RuntimeError("proxy-label alignment failed for one or more review items")
    (out / "blind_manifest.json").write_text(json.dumps({"items": blind}, indent=2))
    (out / "review_data.js").write_text(
        "window.REVIEW_ITEMS = " + json.dumps(blind, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    (out / "revealed_proxy_labels.json").write_text(json.dumps({"items": revealed}, indent=2))
    summary = {
        "n_items": len(blind),
        "n_tasks": len(candidates),
        "per_task": args.per_task,
        "blind_file": str(out / "blind_manifest.json"),
        "revealed_file": str(out / "revealed_proxy_labels.json"),
        "warning": "Review labels are human completion judgments, not verified task success.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
