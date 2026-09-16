"""Pack one episode's Omega clips into candidate-wise risk samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="episode directory containing variant dirs")
    ap.add_argument("--output", required=True)
    ap.add_argument("--variants", nargs="+", default=[
        "expert", "joint_noise_0.08", "joint_noise_0.16",
        "joint_noise_0.28", "gripper_delay"])
    args = ap.parse_args()
    root, out = Path(args.input), Path(args.output)
    files = {v: sorted((root / v).glob("*.npz")) for v in args.variants}
    if any(not fs for fs in files.values()):
        raise SystemExit({v: len(fs) for v, fs in files.items()})
    by_name = {v: {f.name: f for f in fs} for v, fs in files.items()}
    names = sorted(set.intersection(*(set(m) for m in by_name.values())))
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        records = [np.load(by_name[v][name]) for v in args.variants]
        ref = records[0]
        actions = np.stack([r["action_chunk"] for r in records]).astype(np.float32)
        np.savez_compressed(
            out / name,
            points=ref["points"].astype(np.float32),
            point_confidence=ref.get("point_confidence", np.ones(ref["points"].shape[:2], np.float32)),
            robot_state=ref["robot_state"].astype(np.float32),
            actions=actions.reshape(len(records), -1),
            action_chunk=actions.reshape(len(records), -1),
            future_points=np.concatenate([r["future_points"] for r in records], axis=0).astype(np.float32),
            affordance=np.concatenate([r["affordance"] for r in records], axis=0).astype(np.float32),
            success=np.concatenate([r["success"] for r in records], axis=0).astype(np.float32),
            collision=np.concatenate([r["collision"] for r in records], axis=0).astype(np.float32),
            candidate_names=np.asarray(args.variants),
            episode=np.asarray(root.name),
        )
    print(f"packed {len(names)} windows, M={len(args.variants)} -> {out}")


if __name__ == "__main__":
    main()
