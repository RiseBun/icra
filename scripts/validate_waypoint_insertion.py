"""Pilot native-waypoint perturbations for peg insertion.

Unlike the failed IK candidate generator, this script starts from the exact
expert action sequence.  A short insertion window is perturbed in joint space
and the nominal expert suffix is replayed afterwards.  The purpose is to
measure controllability before building a factor dataset.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_peg_grasp_events import expert_actions, find_grasp_boundary, make_environment
from peg_insertion_events import profile_demo
from validate_static_ad0 import shape, get_snapshot


def target_from_color(variation):
    from rlbench.const import colors
    expected = np.asarray(colors[int(variation)][1], np.float32)
    cols = np.stack([np.asarray(shape(f"pillar{i}").get_color(), np.float32)
                     for i in range(3)])
    return int(np.argmin(np.linalg.norm(cols - expected[None], axis=1)))


def replay_candidate(task, demo, expert, start, window, perturb, target):
    _, obs = task.reset_to_demo(demo)
    trace = []
    rewards = []
    actions = expert.copy()
    end = min(len(actions), start + window)
    actions[start:end, :7] += perturb[:end-start]
    for i, action in enumerate(actions):
        obs, reward, terminate = task.step(action)
        rewards.append(float(reward))
        trace.append(get_snapshot(task, target))
        if terminate:
            break
    stable = sum(1 for s in trace[-8:] if s["target_xy"] < 0.035)
    final = trace[-1] if trace else None
    return {
        "reward_success": int(any(r > 0.5 for r in rewards)),
        "stable_target_steps": int(stable),
        "final": final,
        "perturb_l2": float(np.linalg.norm(perturb)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--output", default="data/diagnostics/waypoint_insertion_gate")
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--noise", type=float, default=0.008)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--offset-from-boundary", type=int, default=-8,
                    help="window start = target boundary + offset")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in range(args.episodes):
        env, task = make_environment(args.image_size)
        try:
            task.sample_variation()
            variation = int(getattr(task, "_variation_number", 0))
            demo = task.get_demos(1, live_demos=True, max_attempts=10)[0]
            target = target_from_color(variation)
            expert = expert_actions(demo)
            profile = profile_demo(task, demo, expert)
            boundary = profile["boundaries"].get("target_approach_boundary")
            if boundary is None:
                rows.append({"episode": episode, "variation": variation,
                             "error": "no_target_approach"}); continue
            # Keep the perturbation inside the approach segment, before the
            # expert release.  Use a fixed direction per episode and paired
            # signs so action magnitude is identical.
            start = max(0, int(boundary) + args.offset_from_boundary)
            if start >= int(boundary):
                start = max(0, int(boundary) - args.window)
            rng = np.random.default_rng(2027 + episode)
            direction = rng.normal(size=(args.window, 7)).astype(np.float32)
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
            perturb = direction * float(args.noise)
            candidates = {}
            for name, sign in (("nominal", 0.0), ("positive", 1.0), ("negative", -1.0)):
                p = np.zeros_like(perturb) if sign == 0 else perturb * sign
                candidates[name] = replay_candidate(task, demo, expert, start,
                                                     args.window, p, target)
            row = {"episode": episode, "variation": variation, "target": target,
                   "target_approach_boundary": int(boundary), "window_start": start,
                   "window": args.window, "noise": args.noise,
                   "candidates": candidates}
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)
        except Exception as exc:
            row = {"episode": episode, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row); print(json.dumps(row), flush=True)
        finally:
            env.shutdown()
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
