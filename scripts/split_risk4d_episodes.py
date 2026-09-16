"""Create 70/10/10/10 episode-level train/val/calibration/test links."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def variation_flags(episode: Path) -> tuple[bool, bool]:
    """Return (success variation, collision variation) for one episode."""
    successes, collisions = [], []
    for clip in sorted(episode.glob("*.npz")):
        with np.load(clip, allow_pickle=True) as data:
            successes.extend(np.asarray(data["success"]).reshape(-1).tolist())
            collisions.extend(np.asarray(data["collision"]).reshape(-1).tolist())
    return len(set(successes)) > 1, len(set(collisions)) > 1


def has_risk_variation(episode: Path) -> bool:
    return any(variation_flags(episode))


def _ensure_predicate(partitions: dict[str, list[Path]], target: str,
                      predicate) -> None:
    """Swap one episode into target when its metric would otherwise be constant."""
    if any(predicate(episode) for episode in partitions[target]):
        return
    donors = [name for name in ("validation", "calibration", "test", "train")
              if name != target]
    for donor in donors:
        candidate = next((episode for episode in partitions[donor] if predicate(episode)), None)
        if candidate is None:
            continue
        replacement = next((episode for episode in partitions[target]
                            if not predicate(episode)), None)
        if replacement is None:
            return
        partitions[donor].remove(candidate)
        partitions[donor].append(replacement)
        partitions[target].remove(replacement)
        partitions[target].append(candidate)
        return


def stratified_partitions(episodes: list[Path], rng: np.random.Generator) -> dict[str, list[Path]]:
    """Allocate risk-informative episodes to every non-training split first.

    Splitting remains episode-level, but pure random allocation can produce a
    test set with one label only.  Reserving varied episodes first keeps AUROC,
    calibration, and selection metrics identifiable without sharing clips.
    """
    count = len(episodes)
    val_count = max(1, round(0.1 * count))
    calibration_count = max(1, round(0.1 * count))
    test_count = max(1, round(0.1 * count))
    train_count = count - val_count - calibration_count - test_count
    shuffled = [episodes[index] for index in rng.permutation(count)]
    varied = [episode for episode in shuffled if has_risk_variation(episode)]
    plain = [episode for episode in shuffled if not has_risk_variation(episode)]
    held_out: dict[str, list[Path]] = {"validation": [], "calibration": [], "test": []}
    # Give each held-out split one informative episode whenever available.
    for split in ("validation", "calibration", "test"):
        if varied:
            held_out[split].append(varied.pop(0))
    train_seed = varied.pop(0) if varied else None
    remaining = varied + plain
    remaining = [remaining[index] for index in rng.permutation(len(remaining))]
    targets = {"validation": val_count, "calibration": calibration_count, "test": test_count}
    for split in ("validation", "calibration", "test"):
        need = targets[split] - len(held_out[split])
        held_out[split].extend(remaining[:need])
        del remaining[:need]
    train = ([] if train_seed is None else [train_seed]) + remaining
    if len(train) != train_count:
        raise RuntimeError(f"stratified split produced {len(train)} train episodes, expected {train_count}")
    partitions = {"train": train, **held_out}
    # Collision positives are scarce in the current RLBench batch. Keep them
    # in both training and test so collision risk is learnable and testable.
    _ensure_predicate(partitions, "train", lambda episode: variation_flags(episode)[1])
    _ensure_predicate(partitions, "test", lambda episode: variation_flags(episode)[1])
    # Every held-out split needs at least one non-constant success/risk target
    # whenever the collected episodes make that possible.
    for split in ("validation", "calibration", "test"):
        _ensure_predicate(partitions, split, has_risk_variation)
    return partitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    episodes = sorted(path for path in source.glob("episode_*") if path.is_dir())
    if len(episodes) < 10:
        raise ValueError("at least 10 episodes are required for a 70/10/10/10 split")
    rng = np.random.default_rng(args.seed)
    partitions = stratified_partitions(episodes, rng)
    manifest = {"seed": args.seed, "source": str(source), "splits": {}}
    for split, split_episodes in partitions.items():
        split_root = output / split
        split_root.mkdir(parents=True, exist_ok=True)
        manifest["splits"][split] = []
        for episode in split_episodes:
            manifest["splits"][split].append(episode.name)
            for clip in sorted(episode.glob("*.npz")):
                link = split_root / f"{episode.name}_{clip.name}"
                if link.exists() or link.is_symlink():
                    if link.resolve() != clip.resolve():
                        raise FileExistsError(f"refusing to replace {link}")
                    continue
                os.symlink(clip, link)
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: len(value) for key, value in partitions.items()}, indent=2))


if __name__ == "__main__":
    main()
