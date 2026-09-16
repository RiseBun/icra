"""Synthetic trainer for validating the trainable 4D head.

This is not a research dataset. It checks that the frozen-backbone head can be
trained on a multi-GPU workstation before RLBench data preparation begins.
It supports torchrun but also works as a single-process command.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel, action4d_loss


def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    return world, rank, local


def batch(device, b, k, n, m, h):
    base = torch.randn(b, n, 3, device=device)
    velocity = 0.01 * torch.randn(b, n, 3, device=device)
    t = torch.arange(k, device=device).view(1, k, 1, 1)
    points = base[:, None] + t * velocity[:, None]
    actions = torch.randn(b, m, 10, device=device)
    state = torch.randn(b, 16, device=device)
    action_delta = 0.02 * actions[..., :3]
    future = action_delta[:, :, None, None, :] + velocity[:, None, None, :, :]
    future = future.expand(-1, -1, h, -1, -1)
    future = future * torch.arange(1, h + 1, device=device).view(1, 1, h, 1, 1)
    affordance = (points[:, -1, None, :, 2] > 0).float().expand(-1, m, -1)
    success = (actions[..., :3].square().sum(-1) < 1.0).float()
    collision = (actions[..., :3].square().sum(-1) > 4.0).float()
    return points, actions, state, future, affordance, success, collision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--output", default="results/synthetic_action4d.pt")
    args = ap.parse_args()
    world, rank, local = setup_dist()
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    model = ActionConditioned4DModel(
        robot_state_dim=16, action_dim=10, future_steps=8,
        hidden_dim=256, layers=4, heads=8,
    ).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    for step in range(args.steps):
        values = batch(device, args.batch_size, 8, args.points, 8, 8)
        points, actions, state, future, affordance, success, collision = values
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(points, actions, state)
            losses = action4d_loss(out, future, affordance, success, collision)
        opt.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if rank == 0 and (step == 0 or (step + 1) % 5 == 0):
            print(f"step={step + 1} loss={losses['total'].item():.4f}", flush=True)
    if rank == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw = model.module if world > 1 else model
        torch.save({"model": raw.state_dict(), "steps": args.steps}, out_path)
        print(f"saved {out_path}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
