"""Train one independently seeded member of the risk-conditioned 4D ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import RiskConditioned4DModel, risk4d_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Risk4DDataset(args.data)
    probe = dataset[0]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    config = {
        "point_feature_dim": probe["point_features"].shape[-1],
        "robot_state_dim": probe["robot_state"].shape[-1],
        "task_embedding_dim": probe["task_embedding"].shape[-1],
        "hidden_dim": args.hidden_dim,
        "future_steps": probe["future_point_flow"].shape[1],
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
    }
    model = RiskConditioned4DModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    input_keys = ("points", "point_features", "robot_state", "robot_action_flow", "task_embedding")
    target_keys = ("future_point_flow", "task_contact_map", "harmful_collision_map", "success", "collision")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(*(batch[key] for key in input_keys))
                losses = risk4d_loss(output, *(batch[key] for key in target_keys))
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(losses["total"].detach())
        print(f"epoch={epoch + 1} loss={running / len(loader):.6f}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": config, "seed": args.seed}, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
