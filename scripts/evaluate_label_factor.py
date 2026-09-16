"""Pure-label factor diagnostic with one immutable RLBench scene.

One demo and one set of d0/d1/d2 actions are generated. Each action is replayed
without any scene mutation; the same physical endpoint is then evaluated under
three target labels. This isolates the action-to-pillar relation and avoids
RLBench shape swapping entirely. It is a diagnostic, not a task-success claim
unless the target identity is observable in RGB or the instruction.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_peg_grasp_events import expert_actions, make_environment, find_grasp_boundary
from validate_static_ad0 import shape, get_snapshot, replay_to_grasp, build_d0, execute_and_trace


def classify(snapshot, target, threshold=0.035):
    if snapshot["nearest_xy"] > threshold:
        return "drop"
    return "success" if snapshot["nearest"] == target else "wrong_target"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/diagnostics/label_factor")
    ap.add_argument("--image-size", type=int, default=96)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env, task = make_environment(args.image_size)
    rows = []
    try:
        task.sample_variation(); task.reset()
        demo = task.get_demos(1, live_demos=True, max_attempts=5)[0]
        expert = expert_actions(demo)
        boundary = find_grasp_boundary(task, demo, expert)
        if boundary is None:
            raise RuntimeError("no grasp boundary")
        settle = boundary + 6
        replay_to_grasp(task, demo, expert, settle)
        pillars = [shape(f"pillar{i}") for i in range(3)]
        frozen_positions = [np.asarray(p.get_position(), dtype=np.float64).tolist() for p in pillars]
        # Build all actions from exactly one immutable grasp state.
        actions = {i: build_d0(task, i)[0] for i in range(3)}
        profiles = {
            str(i): {
                "joint_l2": float(np.linalg.norm(np.diff(a[:, :7], axis=0), axis=-1).sum()),
                "length": int(len(a)),
            } for i, a in actions.items()
        }
        for direction_id, chunk in actions.items():
            replay_to_grasp(task, demo, expert, settle)
            result = execute_and_trace(task, chunk, target=0)
            endpoint = result["final"]
            for target in range(3):
                rows.append({
                    "direction_id": direction_id,
                    "target_label": target,
                    "label": classify(endpoint, target),
                    "nearest_pillar": endpoint["nearest"],
                    "nearest_xy": endpoint["nearest_xy"],
                    "stable_target_steps": result["stable_target_steps"],
                    "reward_success_sensor": result["reward_success"],
                    "endpoint": endpoint,
                })
        payload = {"frozen_positions": frozen_positions, "profiles": profiles, "rows": rows}
        (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, separators=(",", ":")), flush=True)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
