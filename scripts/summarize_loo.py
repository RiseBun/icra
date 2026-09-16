"""Summarize leave-one-episode-out action-selection results with bootstrap CIs."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/loo_action_results.txt")
    ap.add_argument("--output", default="results/loo_action_summary.txt")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--samples", type=int, default=10000)
    args = ap.parse_args()
    rows = [ast.literal_eval(line) for line in Path(args.input).read_text().splitlines()
            if line.startswith("{")]
    rng = np.random.default_rng(args.seed)
    names = ["fixed_expert_collision_rate", "random_collision_rate",
             "risk_driven_collision_rate", "oracle_collision_rate", "collision_auroc"]
    out = []
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        boot = values[rng.integers(0, len(values), size=(args.samples, len(values)))].mean(1)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        out.append({"metric": name, "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values, ddof=1)),
                    "ci95": [float(lo), float(hi)],
                    "weighted": float(sum(row[name] * row["clips"] for row in rows) /
                                       sum(row["clips"] for row in rows))})
    fixed = np.asarray([row["fixed_expert_collision_rate"] for row in rows])
    risk = np.asarray([row["risk_driven_collision_rate"] for row in rows])
    delta = fixed - risk
    boot_delta = delta[rng.integers(0, len(delta), size=(args.samples, len(delta)))].mean(1)
    out.append({"metric": "risk_reduction_vs_fixed", "mean": float(delta.mean()),
                "std": float(delta.std(ddof=1)),
                "ci95": [float(np.percentile(boot_delta, 2.5)),
                         float(np.percentile(boot_delta, 97.5))],
                "weighted": float(sum((row["fixed_expert_collision_rate"] - row["risk_driven_collision_rate"]) * row["clips"] for row in rows) /
                                   sum(row["clips"] for row in rows))})
    Path(args.output).write_text("\n".join(str(item) for item in out) + "\n")
    print("\n".join(str(item) for item in out))


if __name__ == "__main__":
    main()
