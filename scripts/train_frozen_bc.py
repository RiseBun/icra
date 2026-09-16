"""Train the frozen stochastic BC policy used only to propose action candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policies.frozen_bc import FrozenBCPolicy


@dataclass
class Demo:
    root: Path
    robot: np.ndarray
    joints: np.ndarray
    actions: np.ndarray


def load_demos(root: Path) -> list[Demo]:
    demos = []
    for path in sorted(root.glob("**/trajectory.npz")):
        with np.load(path, allow_pickle=False) as archive:
            semantics = str(archive.get("action_semantics", "missing"))
            if semantics != "absolute_joint_position":
                continue
            demos.append(Demo(
                path.parent,
                np.asarray(archive["low_dim"], np.float32),
                np.asarray(archive["joint_history"], np.float32),
                np.asarray(archive["actions"], np.float32),
            ))
    if len(demos) < 2:
        raise ValueError(
            f"need at least two unambiguous demonstrations under {root}; found {len(demos)}")
    return demos


def padded_history(value: np.ndarray, end: int, length: int) -> np.ndarray:
    start = max(0, end - length + 1)
    result = value[start:end + 1]
    if len(result) < length:
        result = np.concatenate((np.repeat(result[:1], length - len(result), axis=0), result))
    return result


def padded_future(value: np.ndarray, start: int, length: int) -> np.ndarray:
    result = value[start:start + length]
    if len(result) < length:
        result = np.concatenate((result, np.repeat(result[-1:], length - len(result), axis=0)))
    return result


class BCDataset(Dataset):
    def __init__(self, demos: list[Demo], history: int, horizon: int, image_size: int):
        self.demos, self.history, self.horizon, self.image_size = demos, history, horizon, image_size
        self.windows = [(demo_id, step) for demo_id, demo in enumerate(demos)
                        for step in range(len(demo.actions))]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        demo_id, step = self.windows[index]
        demo = self.demos[demo_id]
        image = Image.open(demo.root / f"frame_{step:04d}.png").convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, np.uint8).copy().transpose(2, 0, 1)
        return {
            "rgb": torch.from_numpy(rgb).float() / 255.0,
            "robot": torch.from_numpy(padded_history(demo.robot, step, self.history).copy()),
            "joints": torch.from_numpy(padded_history(demo.joints, step, self.history).copy()),
            "target": torch.from_numpy(padded_future(demo.actions, step, self.horizon).copy()),
        }

    def statistics(self):
        proprio, targets = [], []
        for demo_id, step in self.windows:
            demo = self.demos[demo_id]
            robot = padded_history(demo.robot, step, self.history)
            joints = padded_history(demo.joints, step, self.history)
            proprio.append(np.concatenate((robot, joints), axis=-1).reshape(-1))
            targets.append(padded_future(demo.actions, step, self.horizon)[..., :7])
        proprio = torch.from_numpy(np.stack(proprio)).float()
        targets = torch.from_numpy(np.concatenate(targets, axis=0)).float()
        return proprio.mean(0), proprio.std(0), targets.min(0).values, targets.max(0).values


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for batch in loader:
        output = model(batch["rgb"].to(device), batch["robot"].to(device),
                       batch["joints"].to(device))
        loss = model.loss(output, batch["target"].to(device))
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.detach()) * len(batch["rgb"])
        count += len(batch["rgb"])
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    demos = load_demos(Path(args.data))
    generator = np.random.default_rng(args.seed)
    order = generator.permutation(len(demos))
    val_count = max(1, int(round(len(demos) * args.val_fraction)))
    val_ids, train_ids = order[:val_count], order[val_count:]
    if len(train_ids) == 0:
        raise ValueError("demo-level split left no training demonstrations")
    train = BCDataset([demos[i] for i in train_ids], args.history, args.horizon, args.image_size)
    val = BCDataset([demos[i] for i in val_ids], args.history, args.horizon, args.image_size)
    state_dim = train.demos[0].robot.shape[-1]
    if any(demo.robot.shape[-1] != state_dim for demo in demos):
        raise ValueError("robot state dimension differs across demonstrations")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = {
        "history_steps": args.history, "state_dim": state_dim,
        "action_horizon": args.horizon, "image_size": args.image_size,
        "hidden_dim": args.hidden_dim,
    }
    model = FrozenBCPolicy(**config).to(device)
    model.set_statistics(*(value.to(device) for value in train.statistics()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=2,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, args.batch_size, shuffle=False, num_workers=2,
                            pin_memory=device.type == "cuda")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        with torch.inference_mode():
            val_loss = run_epoch(model, val_loader, device)
        print(f"epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f}", flush=True)
        if val_loss < best:
            best = val_loss
            torch.save({
                "model": model.state_dict(), "model_config": config,
                "action_semantics": "absolute_joint_position",
                "train_demo_ids": train_ids.tolist(), "validation_demo_ids": val_ids.tolist(),
                "best_validation_loss": best, "seed": args.seed,
            }, output)
    print(f"saved frozen BC policy to {output} best_val={best:.5f}")


if __name__ == "__main__":
    main()
