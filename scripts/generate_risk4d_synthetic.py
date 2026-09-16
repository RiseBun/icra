"""Generate structurally valid v2 clips for tests only, never paper metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_risk4d/train")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--action-steps", type=int, default=32)
    parser.add_argument("--future", type=int, default=8)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--robot-points", type=int, default=128)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--robot-state-dim", type=int, default=16)
    parser.add_argument("--task-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for sample_index in range(args.samples):
        points = rng.normal(0, 0.25, (args.history, args.points, 3)).astype(np.float32)
        points[..., 2] = np.abs(points[..., 2])
        confidence = rng.uniform(0.5, 1.0, (args.history, args.points, 1)).astype(np.float32)
        robot_flow = rng.normal(
            0, 0.025, (args.candidates, args.action_steps, args.robot_points, 3)
        ).astype(np.float32).cumsum(axis=1)
        action_strength = np.linalg.norm(robot_flow[:, -1], axis=-1).mean(-1)
        collision = (action_strength > np.median(action_strength)).astype(np.float32)
        success = (1.0 - collision).astype(np.float32)
        future_flow = np.zeros((args.candidates, args.future, args.points, 3), np.float32)
        contact = np.zeros((args.candidates, args.future, args.points), np.float32)
        harmful = np.zeros_like(contact)
        for candidate in range(args.candidates):
            direction = robot_flow[candidate, -1].mean(0)
            for step in range(args.future):
                scale = float(step + 1) / args.future
                future_flow[candidate, step] = direction[None] * scale
            distance = np.linalg.norm(points[-1] - robot_flow[candidate, -1].mean(0), axis=-1)
            nearest = np.argsort(distance)[:max(1, args.points // 16)]
            contact[candidate, :, nearest] = 1.0
            if collision[candidate] > 0:
                harmful[candidate, :, nearest] = 1.0
        view_reduction = rng.normal(0.1, 0.04, args.views).clip(-0.1, 0.3).astype(np.float32)
        view_points = np.repeat(points[None], args.views, axis=0)
        view_points += rng.normal(0, 0.01, view_points.shape).astype(np.float32)
        view_features = rng.uniform(
            0.4, 1.0, (args.views, args.history, args.points, 1)
        ).astype(np.float32)
        candidate_actions = rng.normal(
            0, 0.1, (args.candidates, args.action_steps, 8)
        ).astype(np.float32)
        candidate_actions[..., 7] = rng.integers(
            0, 2, (args.candidates, args.action_steps))
        np.savez_compressed(
            output / f"sample_{sample_index:05d}.npz",
            points=points,
            point_features=confidence,
            robot_state=rng.normal(size=(args.history, args.robot_state_dim)).astype(np.float32),
            candidate_actions=candidate_actions,
            current_joints=np.zeros(8, np.float32),
            robot_action_flow=robot_flow,
            candidate_views=rng.normal(0, 0.1, (args.views, 6)).astype(np.float32),
            view_cost=rng.uniform(0, 1, args.views).astype(np.float32),
            view_safety_cost=rng.uniform(0, 0.1, args.views).astype(np.float32),
            view_points=view_points, view_point_features=view_features,
            task_embedding=rng.normal(size=args.task_dim).astype(np.float32),
            future_point_flow=future_flow,
            task_contact_map=contact,
            harmful_collision_map=harmful,
            success=success,
            collision=collision,
            view_risk_reduction=view_reduction,
        )
    print(f"wrote {args.samples} synthetic test clips to {output}")


if __name__ == "__main__":
    main()
