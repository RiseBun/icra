"""Measure controllability of the native expert insertion suffix."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_peg_grasp_events import expert_actions, make_environment, find_grasp_boundary
from validate_static_ad0 import shape, replay_to_grasp, execute_and_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variations", type=int, nargs="+", default=(0, 2, 4))
    ap.add_argument("--output", default="data/diagnostics/expert_suffix")
    ap.add_argument("--image-size", type=int, default=96)
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in args.variations:
        env, task = make_environment(args.image_size)
        try:
            task.set_variation(v)
            demo = task.get_demos(1, live_demos=True, max_attempts=10)[0]
            from rlbench.const import colors
            expected = np.asarray(colors[v][1], dtype=np.float32)
            cols = np.stack([np.asarray(shape(f"pillar{i}").get_color(), dtype=np.float32) for i in range(3)])
            target = int(np.argmin(np.linalg.norm(cols - expected[None], axis=1)))
            expert = expert_actions(demo)
            boundary = find_grasp_boundary(task, demo, expert)
            if boundary is None:
                rows.append({"variation": v, "error": "no_grasp_boundary"}); continue
            settle = boundary + 6
            replay_to_grasp(task, demo, expert, settle)
            suffix = np.asarray(expert[settle:], dtype=np.float32)
            replay_to_grasp(task, demo, expert, settle)
            result = execute_and_trace(task, suffix, target)
            rows.append({"variation": v, "target": target, "target_color": expected.tolist(),
                         "suffix_length": int(len(suffix)),
                         "reward_success": result["reward_success"],
                         "stable_target_steps": result["stable_target_steps"],
                         "final": result["final"]})
            print(json.dumps(rows[-1], separators=(",", ":")), flush=True)
        except Exception as exc:
            rows.append({"variation": v, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(rows[-1]), flush=True)
        finally:
            env.shutdown()
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
