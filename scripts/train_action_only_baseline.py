"""Train and evaluate an action-only shortcut baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_paper_metrics import auroc, ece
from scripts.train_dataset import ClipDataset


class ActionOnly(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * action_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        relative = actions - actions.mean(dim=1, keepdim=True)
        return self.net(torch.cat((actions, relative), dim=-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = ClipDataset(args.train)
    test = ClipDataset(args.test)
    action_dim = int(train[0]["actions"].shape[-1])
    model = ActionOnly(action_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    model.train()
    for _ in range(args.epochs):
        for item in loader:
            actions = item["actions"].to(device)
            target = torch.stack((item["success"], item["collision"]), dim=-1).to(device)
            logits = model(actions)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    success_prob, collision_prob, success, collision = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for item in DataLoader(test, batch_size=args.batch_size):
            logits = model(item["actions"].to(device)).cpu()
            success_prob.append(torch.sigmoid(logits[..., 0]))
            collision_prob.append(torch.sigmoid(logits[..., 1]))
            success.append(item["success"])
            collision.append(item["collision"])
    sp = torch.cat(success_prob).numpy()
    cp = torch.cat(collision_prob).numpy()
    s = torch.cat(success).numpy()
    c = torch.cat(collision).numpy()
    predicted_risk = (1.0 - sp) + cp
    true_utility = (1.0 - s) + c
    selected = predicted_risk.argmin(1)
    rows = np.arange(len(test))

    result = {
        "model": "action_only_relative_mlp",
        "train_clips": len(train),
        "test_clips": len(test),
        "failure_auroc": auroc((1.0 - sp).reshape(-1), (1.0 - s).reshape(-1)),
        "collision_auroc": auroc(cp.reshape(-1), c.reshape(-1)),
        "success_brier": float(np.mean((sp - s) ** 2)),
        "success_ece": ece(sp.reshape(-1), s.reshape(-1)),
        "fixed_nominal_success": float(s[:, 0].mean()),
        "random_expected_success": float(s.mean(1).mean()),
        "selected_success": float(s[rows, selected].mean()),
        "selected_collision": float(c[rows, selected].mean()),
        "selected_utility": float(true_utility[rows, selected].mean()),
        "selection": selected.tolist(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
