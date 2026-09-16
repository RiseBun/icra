"""Collect RGB-visible target-color episodes without scene mutation."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_peg_grasp_events import expert_actions, make_environment, find_grasp_boundary
from peg_geometry_signals import extract_peg_signals
from validate_static_ad0 import shape, replay_to_grasp, build_d0, execute_and_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-color", type=int, default=1)
    ap.add_argument("--variations", type=int, nargs="+", default=(0, 2, 4))
    ap.add_argument("--output", default="data/diagnostics/native_color_factor")
    ap.add_argument("--image-size", type=int, default=96)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variation in args.variations:
        for rep in range(args.episodes_per_color):
            env, task = make_environment(args.image_size)
            try:
                task.set_variation(int(variation))
                demo = task.get_demos(1, live_demos=True, max_attempts=10)[0]
                # Capture target identity immediately after demo generation;
                # later reset_to_demo() calls may re-run init_episode and pick
                # a different random pillar.
                from rlbench.const import colors
                expected_color = np.asarray(colors[int(variation)][1], dtype=np.float32)
                all_colors = np.stack([np.asarray(shape(f"pillar{i}").get_color(), dtype=np.float32)
                                       for i in range(3)])
                target = int(np.argmin(np.linalg.norm(all_colors - expected_color[None], axis=1)))
                target_color = all_colors[target].tolist()
                expert = expert_actions(demo)
                boundary = find_grasp_boundary(task, demo, expert)
                if boundary is None:
                    rows.append({"variation": variation, "rep": rep, "error": "no_grasp_boundary"})
                    continue
                settle = boundary + 6
                replay_to_grasp(task, demo, expert, settle)
                # Keep the RGB evidence from the grasp state. This is the
                # observable target signal used by the formal predictor.
                _, grasp_obs = task.reset_to_demo(demo)
                history = [grasp_obs]
                for action in expert[:settle]:
                    grasp_obs, _, term = task.step(action)
                    history.append(grasp_obs)
                    if term:
                        break
                history = history[-8:]
                history_rgb = np.stack([o.front_rgb for o in history]).astype(np.uint8)
                npz_name = f"variation_{variation:02d}_rep_{rep:03d}.npz"
                np.savez_compressed(out / npz_name,
                                    rgb_history=history_rgb,
                                    target_color=np.asarray(target_color, dtype=np.float32),
                                    target_pillar=np.int64(target))
                # Build all direction actions from the same grasp state; no
                # pillar or color is changed during any replay.
                actions = {d: build_d0(task, d)[0] for d in range(3)}
                for direction, chunk in actions.items():
                    replay_to_grasp(task, demo, expert, settle)
                    result = execute_and_trace(task, chunk, target)
                    rows.append({
                        "variation": int(variation), "rep": int(rep),
                        "direction_id": int(direction), "target_pillar": target,
                        "target_color": [float(x) for x in target_color],
                        "action_length": int(len(chunk)),
                        "joint_l2": float(np.linalg.norm(np.diff(chunk[:, :7], axis=0), axis=-1).sum()),
                        "reward_success": int(result["reward_success"]),
                        "stable_target_steps": int(result["stable_target_steps"]),
                        "final": result["final"], "rgb_npz": npz_name,
                    })
                print(json.dumps(rows[-3:], separators=(",", ":")), flush=True)
            except Exception as exc:
                rows.append({"variation": variation, "rep": rep,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(json.dumps(rows[-1]), flush=True)
            finally:
                env.shutdown()
    (out / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
