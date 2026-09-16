"""Audit open-loop candidate controllability from native-color manifests."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--xy-threshold", type=float, default=0.005)
    ap.add_argument("--stable-min", type=int, default=6)
    args = ap.parse_args()
    rows = json.loads(args.manifest.read_text())
    rows = [r for r in rows if "direction_id" in r and "final" in r]
    if not rows:
        raise SystemExit("no replay rows")
    for r in rows:
        r["controllable"] = bool(
            r["final"]["nearest_xy"] <= args.xy_threshold and
            r["stable_target_steps"] >= args.stable_min)
    summary = {
        "n": len(rows),
        "controllable": int(sum(r["controllable"] for r in rows)),
        "controllability_rate": float(np.mean([r["controllable"] for r in rows])),
        "mean_nearest_xy": float(np.mean([r["final"]["nearest_xy"] for r in rows])),
        "median_nearest_xy": float(np.median([r["final"]["nearest_xy"] for r in rows])),
        "max_nearest_xy": float(np.max([r["final"]["nearest_xy"] for r in rows])),
        "reward_success": int(sum(r.get("reward_success", 0) for r in rows)),
        "stable_ge6": int(sum(r["stable_target_steps"] >= args.stable_min for r in rows)),
        "by_direction": {}, "by_variation": {},
    }
    for key, field in (("by_direction", "direction_id"), ("by_variation", "variation")):
        groups = {}
        for r in rows:
            groups.setdefault(str(r[field]), []).append(r)
        for g, rs in groups.items():
            summary[key][g] = {
                "n": len(rs),
                "controllable": int(sum(x["controllable"] for x in rs)),
                "mean_nearest_xy": float(np.mean([x["final"]["nearest_xy"] for x in rs])),
                "mean_joint_l2": float(np.mean([x["joint_l2"] for x in rs])),
                "stable_ge6": int(sum(x["stable_target_steps"] >= args.stable_min for x in rs)),
            }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
