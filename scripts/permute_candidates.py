"""Randomly permute candidate order to remove fixed expert-index shortcuts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.npz")):
        data = dict(np.load(path))
        m = data["actions"].shape[0]
        order = rng.permutation(m)
        for key in ("actions", "full_actions", "future_points", "affordance", "success", "collision"):
            if key in data and data[key].shape[0] == m:
                data[key] = data[key][order]
        data["candidate_permutation"] = order.astype(np.int64)
        np.savez_compressed(output / path.name, **data)
    print(f"permuted {len(list(source.glob('*.npz')))} clips")


if __name__ == "__main__":
    main()
