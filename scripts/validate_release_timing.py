"""Test native expert release-timing counterfactuals."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_peg_grasp_events import expert_actions, make_environment
from peg_insertion_events import profile_demo
from validate_static_ad0 import shape, get_snapshot


def target_from_color(v):
    from rlbench.const import colors
    expected = np.asarray(colors[int(v)][1], np.float32)
    cols = np.stack([np.asarray(shape(f"pillar{i}").get_color(), np.float32)
                     for i in range(3)])
    return int(np.argmin(np.linalg.norm(cols - expected[None], axis=1)))


def release_index(expert):
    vals = np.asarray(expert[:, 7])
    # Demonstrations start with the gripper open.  Release is the first
    # open command *after* a closed interval, not index zero.
    closed = np.where(vals < 0.5)[0]
    if len(closed) == 0:
        return None
    start = int(closed[0])
    idx = np.where(vals[start:] >= 0.5)[0]
    return int(start + idx[0]) if len(idx) else None


def run(task, demo, actions, target):
    _, _ = task.reset_to_demo(demo)
    rewards, trace = [], []
    for action in actions:
        _, reward, terminate = task.step(action)
        rewards.append(float(reward)); trace.append(get_snapshot(task, target))
        if terminate: break
    stable = sum(1 for s in trace[-8:] if s["target_xy"] < 0.035)
    return {"reward_success": int(any(x > 0.5 for x in rewards)),
            "stable_target_steps": int(stable), "final": trace[-1]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--output", default="data/diagnostics/release_timing_gate")
    ap.add_argument("--image-size", type=int, default=96); args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); rows=[]
    for ep in range(args.episodes):
        env, task = make_environment(args.image_size)
        try:
            task.sample_variation(); v=int(getattr(task, "_variation_number", 0))
            demo=task.get_demos(1, live_demos=True, max_attempts=10)[0]
            target=target_from_color(v); expert=expert_actions(demo)
            rix=release_index(expert)
            if rix is None:
                rows.append({"episode":ep,"variation":v,"error":"no_release"}); continue
            candidates={}
            for name, shift in (("nominal",0),("early",-8),("late",8)):
                a=expert.copy(); open_at=max(0,min(len(a)-1,rix+shift)); a[open_at:,7]=1.0
                candidates[name]=run(task,demo,a,target)
            row={"episode":ep,"variation":v,"target":target,"release_index":rix,
                 "candidates":candidates}; rows.append(row)
            print(json.dumps(row,separators=(",",":")),flush=True)
        except Exception as exc:
            row={"episode":ep,"error":f"{type(exc).__name__}: {exc}"}; rows.append(row); print(json.dumps(row),flush=True)
        finally: env.shutdown()
    (out/"results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")

if __name__ == "__main__": main()
