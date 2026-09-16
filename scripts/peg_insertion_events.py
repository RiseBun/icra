"""Insertion-event profiling utilities for ``insert_onto_square_peg``.

The simulator-side signals are used only to choose and audit data-collection
states.  They are never part of the RGB-only model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_peg_grasp_events import expert_actions, make_environment
from scripts.peg_geometry_signals import extract_peg_signals


def detect_event_boundaries(
    target_distance: np.ndarray,
    wrong_target_distance: np.ndarray,
    grasped: np.ndarray,
    *,
    target_threshold: float = 0.035,
) -> dict[str, int | None]:
    """Return first post-grasp target/wrong-target proximity indices."""
    target_distance = np.asarray(target_distance, dtype=np.float32).reshape(-1)
    wrong_target_distance = np.asarray(wrong_target_distance, dtype=np.float32).reshape(-1)
    grasped = np.asarray(grasped, dtype=bool).reshape(-1)
    if not (target_distance.shape == wrong_target_distance.shape == grasped.shape):
        raise ValueError("all event signals must share shape [T]")
    after_grasp = np.maximum.accumulate(grasped)
    target = after_grasp & (target_distance <= target_threshold)
    wrong = after_grasp & (wrong_target_distance <= target_threshold) & ~target
    return {
        "grasp_boundary": next((int(i) for i, value in enumerate(grasped) if value), None),
        "target_approach_boundary": next((int(i) for i, value in enumerate(target) if value), None),
        "wrong_target_boundary": next((int(i) for i, value in enumerate(wrong) if value), None),
    }


def profile_demo(task, demo, expert: np.ndarray) -> dict[str, np.ndarray | dict]:
    """Replay an expert demo and collect per-step insertion signals."""
    task.reset_to_demo(demo)
    signals = []
    grasped = []
    rewards = []
    for action in expert:
        value = extract_peg_signals(task)
        signals.append(value)
        grasped.append(bool(task._robot.gripper.get_grasped_objects()))
        _, reward, terminate = task.step(action)
        rewards.append(float(reward))
        if terminate:
            break
    if signals:
        # Record the post-action state as the final profile sample.
        signals.append(extract_peg_signals(task))
        grasped.append(bool(task._robot.gripper.get_grasped_objects()))
    target = np.asarray([row["target_distance"] for row in signals], np.float32)
    wrong = np.asarray([row["wrong_target_distance"] for row in signals], np.float32)
    height = np.asarray([row["ring_height_above_pillar"] for row in signals], np.float32)
    grasped_array = np.asarray(grasped, dtype=bool)
    boundaries = detect_event_boundaries(target, wrong, grasped_array)
    return {
        "target_distance": target,
        "wrong_target_distance": wrong,
        "ring_height_above_pillar": height,
        "grasped": grasped_array,
        "rewards": np.asarray(rewards, np.float32),
        "boundaries": boundaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/diagnostics/peg_insertion_profiles.jsonl")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for local_episode in range(args.episodes):
            episode = args.episode_offset + local_episode
            env = None
            try:
                env, task = make_environment(args.image_size)
                task.sample_variation()
                demo = task.get_demos(1, live_demos=True, max_attempts=args.max_attempts)[0]
                expert = expert_actions(demo)
                profile = profile_demo(task, demo, expert)
                record = {"episode": episode, "status": "ok",
                          "length": int(len(profile["target_distance"])),
                          "boundaries": profile["boundaries"],
                          "target_distance_min": float(profile["target_distance"].min()),
                          "target_distance_max": float(profile["target_distance"].max()),
                          "wrong_target_distance_min": float(profile["wrong_target_distance"].min()),
                          "height_min": float(profile["ring_height_above_pillar"].min()),
                          "height_max": float(profile["ring_height_above_pillar"].max())}
            except Exception as exc:
                record = {"episode": episode, "status": "error",
                          "error": f"{type(exc).__name__}: {exc}"}
            finally:
                if env is not None:
                    env.shutdown()
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
