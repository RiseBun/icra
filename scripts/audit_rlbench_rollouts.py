"""Audit paired RLBench rollout labels before feature extraction."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def scalar(data, key: str, default: float = 0.0) -> float:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return float(value.reshape(-1)[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="task directory containing episode_* folders")
    ap.add_argument("--expected-variants", nargs="*", default=None)
    args = ap.parse_args()

    episodes = sorted(Path(args.root).glob("episode_*"))
    if not episodes:
        raise SystemExit(f"no episodes under {args.root}")

    accepted = 0
    totals = defaultdict(list)
    for episode in episodes:
        records = {}
        for path in sorted(episode.glob("*/metadata.npz")):
            with np.load(path, allow_pickle=True) as data:
                success = scalar(data, "replay_success", scalar(data, "success"))
                collision = float(np.asarray(data["collision"]).max())
                planner = scalar(data, "planner_demo_success", float("nan"))
            variant = path.parent.name
            records[variant] = (success, collision, planner)
            totals[variant].append((success, collision))

        missing = sorted(set(args.expected_variants or []) - set(records))
        nominal_ok = records.get("expert", (0.0, 0.0, 0.0))[0] > 0.5
        successes = [row[0] for row in records.values()]
        mixed_success = len(set(successes)) > 1
        collision_positive = any(row[1] > 0.5 for row in records.values())
        eligible = nominal_ok and not missing and (mixed_success or collision_positive)
        accepted += int(eligible)
        labels = ", ".join(
            f"{name}:s{int(row[0])}/c{int(row[1])}" for name, row in sorted(records.items()))
        print(f"{episode.name} eligible={int(eligible)} nominal={int(nominal_ok)} "
              f"mixed={int(mixed_success)} collision_positive={int(collision_positive)} "
              f"missing={missing} [{labels}]")

    print(f"summary episodes={len(episodes)} eligible={accepted}")
    for variant in sorted(totals):
        rows = np.asarray(totals[variant], dtype=np.float32)
        print(f"{variant}: n={len(rows)} success_rate={rows[:, 0].mean():.3f} "
              f"collision_rate={rows[:, 1].mean():.3f}")


if __name__ == "__main__":
    main()
