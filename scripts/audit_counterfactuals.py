"""Audit paired counterfactual decision files before training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory containing episode_*/decision_*.npz")
    args = ap.parse_args()
    files = sorted(Path(args.root).glob("episode_*/decision_*.npz"))
    if not files:
        raise SystemExit(f"no decision files under {args.root}")

    success = defaultdict(list)
    collision = defaultdict(list)
    mixed = 0
    all_success = 0
    all_failure = 0
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            names = [str(x) for x in data["candidate_names"]]
            s = np.asarray(data["success"], dtype=np.float32)
            c = np.asarray(data["collision"], dtype=np.float32)
        if len(set(s.tolist())) > 1 or len(set(c.tolist())) > 1:
            mixed += 1
        if np.all(s > 0.5):
            all_success += 1
        if np.all(s <= 0.5):
            all_failure += 1
        for name, sv, cv in zip(names, s, c):
            success[name].append(float(sv))
            collision[name].append(float(cv))

    print(f"decisions={len(files)} mixed_label_decisions={mixed} "
          f"all_success={all_success} all_failure={all_failure}")
    for name in sorted(success):
        s = np.asarray(success[name])
        c = np.asarray(collision[name])
        print(f"{name}: n={len(s)} success_rate={s.mean():.3f} "
              f"collision_rate={c.mean():.3f}")


if __name__ == "__main__":
    main()
