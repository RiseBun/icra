"""Paired L2 sweep for grasp-boundary counterfactuals.

For every episode, one expert demonstration, one grasp boundary, and one
unit-norm joint perturbation direction are reused across all amplitudes.  The
only difference between ``noise_pos`` and ``noise_neg`` is the sign of that
same perturbation.  This makes the success relation a valid cross-episode
test of directional causality rather than a comparison of unrelated demos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_peg_grasp_events import (
    expert_actions,
    find_grasp_boundary,
    make_environment,
)


def _chunk(expert: np.ndarray, start: int, horizon: int) -> np.ndarray:
    chunk = np.asarray(expert[start:start + horizon], dtype=np.float32).copy()
    if len(chunk) == 0:
        raise ValueError("empty expert chunk")
    if len(chunk) < horizon:
        chunk = np.pad(chunk, ((0, horizon - len(chunk)), (0, 0)), mode="edge")
    return chunk


def make_paired_noise_actions(
    expert: np.ndarray,
    start: int,
    horizon: int,
    amplitudes: list[float] | tuple[float, ...],
    rng: np.random.Generator,
    noise_std: float = 0.05,
) -> dict[str, dict[str, np.ndarray]]:
    """Return matched +/- actions for every requested per-step L2 amplitude."""
    nominal = _chunk(expert, start, horizon)
    direction = rng.normal(0.0, noise_std, nominal[:, :7].shape).astype(np.float32)
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction /= np.maximum(norms, 1e-8)
    result: dict[str, dict[str, np.ndarray]] = {}
    for amplitude in amplitudes:
        value = float(amplitude)
        positive = nominal.copy()
        negative = nominal.copy()
        positive[:, :7] += value * direction
        negative[:, :7] -= value * direction
        for action in (positive, negative):
            action[:, 7] = (action[:, 7] >= 0.5).astype(np.float32)
        key = f"{value:.5f}"
        result[key] = {"noise_pos": positive, "noise_neg": negative}
    return result


def replay_outcome(task, demo, expert: np.ndarray, start: int,
                   chunk: np.ndarray, horizon: int) -> dict[str, bool]:
    """Replay one candidate and return only labels needed by the L2 criterion."""
    task.reset_to_demo(demo)
    for action in expert[:start]:
        _, _, terminate = task.step(action)
        if terminate:
            return {"success": False, "grasped": False, "grasp_failure": True}

    grasped = False
    success = False
    for action in chunk[:horizon]:
        _, reward, terminate = task.step(action)
        grasped = grasped or bool(task._robot.gripper.get_grasped_objects())
        success = success or float(reward) > 0.5
        if terminate:
            break
    if not success:
        for action in expert[start + horizon:]:
            _, reward, terminate = task.step(action)
            success = success or float(reward) > 0.5
            if terminate:
                break
    return {"success": bool(success), "grasped": bool(grasped),
            "grasp_failure": not bool(grasped)}


def classify_relation(pos_success: bool, neg_success: bool) -> str:
    if bool(pos_success) != bool(neg_success):
        return "anti_correlated"
    return "both_success" if pos_success else "both_failure"


def summarize_relations(records: list[dict]) -> dict[str, dict[str, float | int]]:
    """Aggregate one relation per episode and amplitude."""
    grouped: dict[str, list[str]] = {}
    for record in records:
        for amplitude, pair in record["amplitudes"].items():
            relation = classify_relation(pair["noise_pos"]["success"],
                                         pair["noise_neg"]["success"])
            grouped.setdefault(amplitude, []).append(relation)
    summary = {}
    for amplitude, relations in sorted(grouped.items(), key=lambda item: float(item[0])):
        counts = {name: relations.count(name) for name in
                  ("anti_correlated", "both_success", "both_failure")}
        total = len(relations)
        summary[amplitude] = {**counts, "total": total,
                              "anti_correlated_ratio": counts["anti_correlated"] / max(total, 1)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/diagnostics/paired_peg_noise_sweep")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--amplitudes", default="0.02,0.04,0.06,0.08")
    parser.add_argument("--offset", type=int, default=-2)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="maximum live-demo planning attempts per episode")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    amplitudes = [float(value) for value in args.amplitudes.split(",") if value.strip()]
    if not amplitudes or any(value <= 0 for value in amplitudes):
        raise ValueError("--amplitudes must contain positive values")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for local_episode in range(args.episodes):
        episode = args.episode_offset + local_episode
        env = None
        try:
            print(json.dumps({"episode": episode, "status": "launching"}), flush=True)
            env, task = make_environment(args.image_size)
            task.sample_variation()
            print(json.dumps({"episode": episode, "status": "generating_demo"}), flush=True)
            demo = task.get_demos(1, live_demos=True, max_attempts=args.max_attempts)[0]
            expert = expert_actions(demo)
            boundary = find_grasp_boundary(task, demo, expert)
            if boundary is None:
                print(json.dumps({"episode": episode, "status": "no_grasp_boundary"}), flush=True)
                continue
            start = boundary + args.offset
            if start < 0 or start >= len(expert):
                print(json.dumps({"episode": episode, "status": "start_out_of_range",
                                  "boundary": boundary, "start": start}), flush=True)
                continue
            rng = np.random.default_rng(args.seed + episode * 10007)
            print(json.dumps({"episode": episode, "status": "replaying",
                              "boundary": int(boundary), "start": int(start)}), flush=True)
            candidates = make_paired_noise_actions(expert, start, args.horizon,
                                                   amplitudes, rng)
            episode_record = {"episode": episode, "boundary": int(boundary),
                              "start": int(start), "offset": int(args.offset),
                              "amplitudes": {}, "status": "ok"}
            for amplitude, pair in candidates.items():
                outcomes = {}
                for name, actions in pair.items():
                    outcomes[name] = replay_outcome(task, demo, expert, start,
                                                    actions, args.horizon)
                outcomes["relation"] = classify_relation(
                    outcomes["noise_pos"]["success"], outcomes["noise_neg"]["success"])
                episode_record["amplitudes"][amplitude] = outcomes
            records.append(episode_record)
            print(json.dumps(episode_record), flush=True)
        except Exception as exc:
            print(json.dumps({"episode": episode, "status": "error",
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        finally:
            if env is not None:
                env.shutdown()

    summary = summarize_relations(records)
    payload = {"config": vars(args), "episodes": records, "summary": summary}
    (output / f"paired_sweep_{args.episode_offset:04d}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "episodes": len(records), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
