"""Train action-only or geometry-scalar v2 risk controls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import ScalarCandidateRiskModel, scalar_risk_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--type", choices=("action_only", "geometry_scalar"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Risk4DDataset(args.data)
    probe = dataset[0]
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=2)
    config = {
        "point_feature_dim": probe["point_features"].shape[-1],
        "robot_state_dim": probe["robot_state"].shape[-1],
        "task_embedding_dim": probe["task_embedding"].shape[-1],
        "hidden_dim": args.hidden_dim, "layers": args.layers, "heads": args.heads,
        "include_geometry": args.type == "geometry_scalar",
    }
    model = ScalarCandidateRiskModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(
                batch["points"], batch["point_features"], batch["robot_state"],
                batch["robot_action_flow"], batch["task_embedding"],
            )
            loss = scalar_risk_loss(output, batch["success"], batch["collision"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
        print(f"epoch={epoch + 1} loss={running / len(loader):.6f}", flush=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "model_config": config,
        "baseline_type": args.type, "seed": args.seed,
    }, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
