"""Train a risk head on top of a frozen PointWorld-like 4D predictor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import (PointWorld4DModel, RiskFrom4DBeliefModel,
                    risk_from_4d_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--perception-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Risk4DDataset(args.data)
    probe = dataset[0]
    perception_state = torch.load(args.perception_checkpoint, map_location="cpu", weights_only=True)
    perception = PointWorld4DModel(**perception_state["model_config"]).to(device)
    perception.load_state_dict(perception_state["model"]); perception.eval()
    for parameter in perception.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    config = {
        "point_feature_dim": probe["point_features"].shape[-1],
        "robot_state_dim": probe["robot_state"].shape[-1],
        "hidden_dim": args.hidden_dim,
        "future_steps": probe["future_point_flow"].shape[1],
        "layers": args.layers, "heads": args.heads, "dropout": args.dropout,
    }
    model = RiskFrom4DBeliefModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                predicted_flow = perception(
                    batch["points"], batch["point_features"],
                    batch["robot_state"], batch["robot_action_flow"],
                ).future_point_flow
            output = model(
                batch["points"], batch["point_features"], batch["robot_state"],
                batch["robot_action_flow"], predicted_flow,
            )
            losses = risk_from_4d_loss(
                output, batch["task_contact_map"], batch["harmful_collision_map"],
                batch["success"], batch["collision"],
            )
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); running += float(losses["total"].detach())
        print(f"epoch={epoch + 1} loss={running / len(loader):.6f}", flush=True)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": config,
                "perception_checkpoint": str(args.perception_checkpoint),
                "baseline_type": "risk_from_frozen_4d", "seed": args.seed}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
