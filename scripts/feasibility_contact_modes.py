"""Feasibility trial for contact-mode prediction from current geometry and action.

The future contact/collision maps are used only to derive supervision.  The
predictor sees current scene statistics, robot state, and candidate robot
surface flow, so this is a leakage-free first test of whether contact modes are
identifiable from the proposed inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch import nn


MODE_NAMES = ("free", "approach", "valid_contact", "sliding_or_grasping",
              "completion", "harmful_collision")


def derive_modes(sample: dict[str, np.ndarray], motion_threshold: float = 1e-3,
                 completion_index: int | None = None) -> np.ndarray:
    contact = np.asarray(sample["task_contact_map"], np.float32) > 0.5
    harmful = np.asarray(sample["harmful_collision_map"], np.float32) > 0.5
    flow = np.asarray(sample["future_point_flow"], np.float32)
    if contact.shape != harmful.shape or flow.shape[:3] != contact.shape:
        raise ValueError("contact, harmful collision, and future flow shapes do not align")
    motion = np.linalg.norm(flow, axis=-1).mean(-1)
    moving = motion > motion_threshold
    # Collapse the spatial maps to one interpretable mode per candidate/time.
    any_contact = contact.any(-1)
    any_harmful = harmful.any(-1)
    modes = np.zeros(contact.shape[:2], np.int64)
    modes[moving & ~any_contact & ~any_harmful] = 1
    modes[any_contact & ~moving & ~any_harmful] = 2
    modes[any_contact & moving & ~any_harmful] = 3
    if completion_index is None:
        completion_index = contact.shape[1] - 1
    success = np.asarray(sample.get("success", np.zeros(contact.shape[0])), np.float32) > 0.5
    modes[success[:, None] & (np.arange(contact.shape[1])[None] == completion_index)] = 4
    modes[any_harmful] = 5
    return modes


def feature_vector(sample: dict[str, np.ndarray], action_only: bool = False) -> np.ndarray:
    points = np.asarray(sample["points"], np.float32)
    state = np.asarray(sample["robot_state"], np.float32)
    flow = np.asarray(sample["robot_action_flow"], np.float32)
    scene = points[-1]
    scene_stats = np.concatenate((scene.mean(0), scene.std(0), scene.min(0), scene.max(0)))
    state_stats = np.concatenate((state[-1], state.mean(0), state.std(0)))
    candidate = np.concatenate((
        flow.mean((1, 2)), flow.std((1, 2)), flow.min((1, 2)), flow.max((1, 2)),
        flow[:, -1].mean(1),
    ), axis=-1)
    context = np.broadcast_to(np.concatenate((scene_stats, state_stats)),
                              (flow.shape[0], scene_stats.size + state_stats.size))
    if action_only:
        return np.concatenate((candidate, np.broadcast_to(state[-1], (flow.shape[0], state.shape[-1]))), -1)
    return np.concatenate((context, candidate), -1)


class ContactModeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, modes: int = 6,
                 horizon: int = 8):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                     nn.GELU())
        self.time = nn.Embedding(horizon, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,M,F], output [B,M,H,C]
        z = self.encoder(x)[:, :, None] + self.time.weight[None, None]
        return self.head(z)


def collect(data: Path, action_only: bool):
    files = sorted(data.glob("**/*.npz"))
    if not files:
        raise ValueError(f"no NPZ files under {data}")
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            sample = {key: np.array(archive[key], copy=True) for key in archive.files}
        if "robot_action_flow" not in sample:
            raise ValueError(f"{path} has no robot_action_flow; run FK export first")
        modes = derive_modes(sample)
        rows.append({"path": str(path), "episode": path.parent.name,
                     "x": feature_vector(sample, action_only), "y": modes})
    return rows


def run(rows, test_episode: str, seed: int, epochs: int, action_only: bool):
    torch.manual_seed(seed)
    train = [r for r in rows if r["episode"] != test_episode]
    test = [r for r in rows if r["episode"] == test_episode]
    if not train or not test:
        raise ValueError(f"split has train={len(train)} test={len(test)}")
    x_train = torch.from_numpy(np.stack([r["x"] for r in train])).float()
    y_train = torch.from_numpy(np.stack([r["y"] for r in train])).long()
    x_test = torch.from_numpy(np.stack([r["x"] for r in test])).float()
    y_test = torch.from_numpy(np.stack([r["y"] for r in test])).long()
    model = ContactModeNet(x_train.shape[-1], horizon=y_train.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    # Weight rare modes but keep the objective finite for a split with absent classes.
    counts = torch.bincount(y_train.reshape(-1), minlength=len(MODE_NAMES)).float()
    weight = (counts.sum() / counts.clamp_min(1)).sqrt()
    for _ in range(epochs):
        logits = model(x_train)
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(MODE_NAMES)), y_train.reshape(-1), weight=weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.inference_mode():
        normal = model(x_test).argmax(-1)
        permutation = torch.randperm(x_test.shape[1], generator=torch.Generator().manual_seed(seed))
        shuffled = model(x_test[:, permutation]).argmax(-1)
        zero = model(torch.zeros_like(x_test)).argmax(-1)
    y_true, y_pred = y_test.numpy().reshape(-1), normal.numpy().reshape(-1)
    labels = list(range(len(MODE_NAMES)))
    result = {
        "test_episode": test_episode, "train_decisions": len(train), "test_decisions": len(test),
        "action_only": action_only, "epochs": epochs,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "shuffle_accuracy": float(accuracy_score(y_true, shuffled.numpy().reshape(-1))),
        "zero_action_accuracy": float(accuracy_score(y_true, zero.numpy().reshape(-1))),
        "shuffle_degradation": float(accuracy_score(y_true, y_pred) - accuracy_score(y_true, shuffled.numpy().reshape(-1))),
        "class_counts_test": np.bincount(y_true, minlength=len(MODE_NAMES)).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "mode_names": list(MODE_NAMES),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-episode", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    rows = collect(Path(args.data), action_only=False)
    episodes = sorted({r["episode"] for r in rows})
    test_episode = args.test_episode or episodes[-1]
    modes = np.concatenate([r["y"].reshape(-1) for r in rows])
    result = {
        "data": args.data, "decisions": len(rows), "episodes": episodes,
        "derived_mode_counts": dict(zip(MODE_NAMES, np.bincount(modes, minlength=len(MODE_NAMES)).tolist())),
        "decision_mode_variation": int(sum(np.unique(r["y"]).size > 1 for r in rows)),
        "full_geometry": run(rows, test_episode, args.seed, args.epochs, False),
        "action_only": run(collect(Path(args.data), action_only=True), test_episode, args.seed, args.epochs, True),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text)


if __name__ == "__main__":
    main()
