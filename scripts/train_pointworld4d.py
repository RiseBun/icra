"""Train the geometry-only PointWorld-like 4D perception adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import PointWorld4DModel, pointworld4d_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Risk4DDataset(args.data)
    probe = dataset[0]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    config = {
        "point_feature_dim": probe["point_features"].shape[-1],
        "robot_state_dim": probe["robot_state"].shape[-1],
        "hidden_dim": args.hidden_dim, "future_steps": probe["future_point_flow"].shape[1],
        "layers": args.layers, "heads": args.heads, "dropout": args.dropout,
    }
    model = PointWorld4DModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                output = model(batch["points"], batch["point_features"],
                               batch["robot_state"], batch["robot_action_flow"])
                losses = pointworld4d_loss(output, batch["future_point_flow"])
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running += float(losses["total"].detach())
        print(f"epoch={epoch + 1} loss={running / len(loader):.6f}", flush=True)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": config, "seed": args.seed,
                "baseline_type": "pointworld4d_perception"}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
