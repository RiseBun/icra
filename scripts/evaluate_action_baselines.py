"""Evaluate deterministic action-selection baselines across episode folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--output", default="results/action_baselines.txt")
    args = ap.parse_args()
    rows = []
    for root in args.roots:
        labels, actions = [], []
        for path in sorted(Path(root).glob("*.npz")):
            data = np.load(path)
            labels.append(data["collision"].astype(np.float32))
            actions.append(data["actions"].astype(np.float32))
        y = np.stack(labels)
        a = np.stack(actions)
        n = np.arange(len(y))
        selectors = {
            "fixed_expert": np.zeros(len(y), dtype=np.int64),
            "min_chunk_l2": np.linalg.norm(a, axis=2).argmin(1),
            "min_chunk_std": np.std(a, axis=2).argmin(1),
            "max_chunk_l2": np.linalg.norm(a, axis=2).argmax(1),
        }
        row = {"root": str(root), "clips": len(y)}
        for name, index in selectors.items():
            row[name] = float(y[n, index].mean())
        rows.append(row)
    print("\n".join(str(row) for row in rows))
    Path(args.output).write_text("\n".join(str(row) for row in rows) + "\n")


if __name__ == "__main__":
    main()
