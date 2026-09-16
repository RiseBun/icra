"""Train risk-reduction prediction on a disjoint active-view split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataio import Risk4DDataset
from models import RiskCalibration, ViewValueModel, load_risk4d_checkpoint, view_value_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--risk-checkpoints", nargs="+", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    risk_models = [load_risk4d_checkpoint(path, device) for path in args.risk_checkpoints]
    calibration = RiskCalibration.load(args.calibration)
    for risk_model in risk_models:
        for parameter in risk_model.parameters():
            parameter.requires_grad_(False)
    dataset = Risk4DDataset(args.data)
    probe = dataset[0]
    for key in ("candidate_views", "view_risk_reduction"):
        if key not in probe:
            raise ValueError(f"view-value training requires {key}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    belief_dim = risk_models[0].hidden_dim
    model = ViewValueModel(belief_dim, belief_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                risks = [risk_model(
                    batch["points"], batch["point_features"], batch["robot_state"],
                    batch["robot_action_flow"], batch["task_embedding"],
                ) for risk_model in risk_models]
                success_logits = torch.stack([risk.success_logits for risk in risks]).mean(0)
                collision_logits = torch.stack([risk.collision_logits for risk in risks]).mean(0)
                probabilities = torch.stack([
                    torch.stack((risk.success_logits.sigmoid(), risk.collision_logits.sigmoid()), -1)
                    for risk in risks
                ])
                epistemic = probabilities.var(0, correction=0).mean(-1)
                action_risk = calibration.risk_bound(
                    success_logits, collision_logits, epistemic,
                )
                belief = torch.stack([risk.belief for risk in risks]).mean(0)
            output = model(belief, batch["candidate_views"], action_risk)
            loss = view_value_loss(output, batch["view_risk_reduction"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
        print(f"epoch={epoch + 1} loss={running / len(loader):.6f}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_config": {"belief_dim": belief_dim, "hidden_dim": belief_dim},
        "risk_checkpoints": args.risk_checkpoints,
        "calibration": args.calibration,
        "seed": args.seed,
    }, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
