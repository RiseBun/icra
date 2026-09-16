"""Pack synchronized perturbation variants into true M-candidate clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def pack(index, variant_paths, output, history=8, future=8, stride=16, full_horizon=128):
    records = [np.load(path) for path in variant_paths]
    reference = records[0]
    future_points = np.concatenate([record["future_points"] for record in records], axis=0)
    full_sequences = []
    for record in records:
        if "source" in record:
            source = str(np.asarray(record["source"]).reshape(-1)[0])
            raw = np.load(source)
            sequence = np.asarray(raw["actions"], dtype=np.float32)[:full_horizon]
            if len(sequence) < full_horizon:
                sequence = np.pad(sequence, ((0, full_horizon - len(sequence)), (0, 0)))
            full_sequences.append(sequence)
    if all("action_chunk" in record for record in records):
        actions = np.stack([record["action_chunk"] for record in records], axis=0)
        actions = actions.reshape(actions.shape[0], -1)
    elif all("source" in record for record in records):
        chunks = []
        for record, path in zip(records, variant_paths):
            source = str(np.asarray(record["source"]).reshape(-1)[0])
            raw = np.load(source)
            start = index * stride
            chunks.append(np.asarray(raw["actions"][start + history:start + history + future], dtype=np.float32))
        actions = np.stack(chunks, axis=0).reshape(len(chunks), -1)
    else:
        actions = np.concatenate([record["actions"] for record in records], axis=0)
    success = np.concatenate([record["success"] for record in records], axis=0)
    collision = np.concatenate([record["collision"] for record in records], axis=0)
    affordance = np.concatenate([record["affordance"] for record in records], axis=0)
    confidence = reference["point_confidence"]
    np.savez_compressed(
        output / f"clip_{index:06d}.npz",
        points=reference["points"], point_confidence=confidence,
        robot_state=reference["robot_state"], actions=actions,
        future_points=future_points, affordance=affordance,
        success=success, collision=collision,
        coordinate_frame=reference.get("coordinate_frame", np.asarray("omega_ref_calibrated")),
        source=np.asarray([str(path) for path in variant_paths]),
        full_actions=np.stack(full_sequences, axis=0).reshape(len(full_sequences), -1)
        if len(full_sequences) == len(records) else np.zeros((len(records), full_horizon * 8), np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/features/multiview_omega_perturbed")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--output", default="data/features/candidates_open_drawer")
    ap.add_argument("--test-count", type=int, default=2)
    ap.add_argument("--variants", nargs="+",
                    default=("expert", "joint_noise_0.03", "gripper_delay"))
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--future", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--full-horizon", type=int, default=128)
    args = ap.parse_args()
    variants = tuple(args.variants)
    roots = [Path(args.input) / variant / args.camera for variant in variants]
    files = [sorted(root.glob("clip_*.npz")) for root in roots]
    count = min(map(len, files))
    if count == 0:
        raise SystemExit("no exported clips found for all variants")
    output = Path(args.output)
    train_root, test_root = output / "train", output / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)
    test_start = max(0, count - args.test_count)
    for i in range(count):
        paths = [file_list[i] for file_list in files]
        pack(i, paths, test_root if i >= test_start else train_root,
             history=args.history, future=args.future, stride=args.stride,
             full_horizon=args.full_horizon)
    print(f"built {test_start} train and {count - test_start} test clips with M={len(variants)}")


if __name__ == "__main__":
    main()
