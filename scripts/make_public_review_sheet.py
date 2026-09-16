"""Make a blind CSV sheet for human progress review.

The CSV contains no proxy labels. Reviewers use the accompanying video/frame
package and assign 1 (clearly progressing), 0 (not progressing/regressing),
or -1 (uncertain). The latter is excluded from agreement metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--raters", nargs="+", default=["rater_1", "rater_2"])
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for item in manifest.get("items", []):
        for rater in args.raters:
            rows.append(
                {
                    "review_id": item["review_id"],
                    "rater": rater,
                    "task": item["task"],
                    "episode_index": item["episode_index"],
                    "timestamp": item["timestamp"],
                    "human_label": "",
                    "confidence": "",
                    "notes": "",
                }
            )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["review_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"n_items": len(manifest.get("items", [])), "n_rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
