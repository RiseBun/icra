"""Train the compact Omega-ActionWorld latent transition on cached NPZ clips."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.omega_actionworld import OmegaActionWorld, omega_actionworld_loss


class CachedLatentDataset(Dataset):
    def __init__(self, root: str | Path) -> None:
        self.files = sorted(Path(root).glob("**/*.npz"))
        if not self.files:
            raise ValueError(f"no cached latent NPZ files found in {root}")
        with np.load(self.files[0], allow_pickle=False) as archive:
            required = ("omega_latent_tokens", "omega_future_latent_tokens",
                        "robot_state", "robot_action_flow")
            missing = [key for key in required if key not in archive]
            if missing:
                raise ValueError(f"cached data is missing {missing}; rerun cache_omega_latents")
            self.probe = {key: np.array(archive[key], copy=True) for key in required}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.files[index], allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in (
                "omega_latent_tokens", "omega_future_latent_tokens",
                "robot_state", "robot_action_flow")}
        # Future targets are [M,H,T,D], while Omega extraction returns [M,H,T,D].
        if arrays["omega_future_latent_tokens"].ndim != 4:
            raise ValueError("omega_future_latent_tokens must have shape [M,H,T,D]")
        return {key: torch.from_numpy(value).float() for key, value in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CachedLatentDataset(args.data)
    probe = dataset.probe
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == "cuda")
    config = {
        "latent_dim": int(probe["omega_latent_tokens"].shape[-1]),
        "robot_state_dim": int(probe["robot_state"].shape[-1]),
        "hidden_dim": args.hidden_dim, "future_steps": int(probe["omega_future_latent_tokens"].shape[1]),
        "layers": args.layers, "heads": args.heads, "dropout": 0.1,
    }
    model = OmegaActionWorld(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
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
                output = model(batch["omega_latent_tokens"], batch["robot_state"],
                               batch["robot_action_flow"])
                losses = omega_actionworld_loss(output, batch["omega_future_latent_tokens"])
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running += float(losses["total"].detach())
        print(f"epoch={epoch + 1} loss={running / len(loader):.6f}", flush=True)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": config,
                "baseline_type": "omega_actionworld", "seed": args.seed}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
