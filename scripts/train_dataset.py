"""Train the action-conditioned head on feature-contract NPZ clips."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.action4d import ActionConditioned4DModel, action4d_loss


class ClipDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).glob("**/*.npz"))
        if not self.files:
            raise ValueError(f"no npz clips found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        d = np.load(self.files[index])
        points = torch.from_numpy(d["points"]).float()
        confidence = torch.from_numpy(d["point_confidence"]).float() if "point_confidence" in d else torch.ones(points.shape[:2])
        future = torch.from_numpy(d["future_points"]).float()
        if "full_actions" in d:
            action_chunk = torch.from_numpy(d["full_actions"]).float()
        else:
            action_chunk = torch.from_numpy(d["action_chunk"]).float() if "action_chunk" in d else torch.from_numpy(d["actions"]).float()
        if action_chunk.ndim > 2:
            action_chunk = action_chunk.flatten(start_dim=1)
        return {
            "points": points,
            "features": confidence.unsqueeze(-1),
            "actions": action_chunk,
            "robot_state": torch.from_numpy(d["robot_state"]).float(),
            "future_delta": future - points[-1][None, None],
            "affordance": torch.from_numpy(d["affordance"]).float(),
            "success": torch.from_numpy(d["success"]).float(),
            "collision": torch.from_numpy(d["collision"]).float(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features/synthetic")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--output", default="results/action4d_dataset.pt")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--absolute-actions", action="store_true",
                    help="disable candidate-relative action encoding")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ClipDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    probe = ds[0]
    model = ActionConditioned4DModel(
        robot_state_dim=int(probe["robot_state"].numel()),
        action_dim=int(probe["actions"].shape[-1]), future_steps=int(probe["future_delta"].shape[1]),
        hidden_dim=args.hidden_dim, layers=args.layers, heads=args.heads,
        point_feature_dim=int(probe["features"].shape[-1]),
        relative_actions=not args.absolute_actions,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for item in loader:
            item = {k: v.to(device) for k, v in item.items()}
            out = model(item["points"], item["actions"], item["robot_state"], item["features"])
            losses = action4d_loss(
                out, item["future_delta"], item["affordance"],
                item["success"], item["collision"],
            )
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(losses["total"].detach())
        print(f"epoch={epoch + 1} loss={total / len(loader):.4f}", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "data": str(args.data),
        "model_config": {
            "robot_state_dim": int(probe["robot_state"].numel()),
            "action_dim": int(probe["actions"].shape[-1]),
            "future_steps": int(probe["future_delta"].shape[1]),
            "hidden_dim": args.hidden_dim, "layers": args.layers, "heads": args.heads,
            "point_feature_dim": int(probe["features"].shape[-1]),
            "relative_actions": not args.absolute_actions,
        },
    }, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
