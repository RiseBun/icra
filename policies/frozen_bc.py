"""Small stochastic behavior-cloning policy for paper candidate generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrozenBCPolicy(nn.Module):
    """Predict a distribution over absolute joint-position action chunks."""

    def __init__(self, history_steps: int, state_dim: int, action_horizon: int,
                 image_size: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.history_steps = history_steps
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        proprio_dim = history_steps * (state_dim + 8)
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.joint_mean = nn.Linear(hidden_dim, action_horizon * 7)
        self.joint_log_std = nn.Linear(hidden_dim, action_horizon * 7)
        self.gripper_logits = nn.Linear(hidden_dim, action_horizon)
        self.register_buffer("proprio_mean", torch.zeros(proprio_dim))
        self.register_buffer("proprio_std", torch.ones(proprio_dim))
        self.register_buffer("joint_min", torch.full((7,), -3.0))
        self.register_buffer("joint_max", torch.full((7,), 3.0))

    def set_statistics(self, proprio_mean: torch.Tensor, proprio_std: torch.Tensor,
                       joint_min: torch.Tensor, joint_max: torch.Tensor) -> None:
        self.proprio_mean.copy_(proprio_mean)
        self.proprio_std.copy_(proprio_std.clamp_min(1e-4))
        self.joint_min.copy_(joint_min)
        self.joint_max.copy_(joint_max)

    def forward(self, rgb: torch.Tensor, robot_history: torch.Tensor,
                joint_history: torch.Tensor) -> dict[str, torch.Tensor]:
        if robot_history.shape[1:] != (self.history_steps, self.state_dim):
            raise ValueError(
                f"robot history has {tuple(robot_history.shape[1:])}, expected "
                f"{(self.history_steps, self.state_dim)}")
        if joint_history.shape[1:] != (self.history_steps, 8):
            raise ValueError("joint history shape does not match checkpoint")
        visual = self.visual(rgb)
        proprio = torch.cat((robot_history, joint_history), dim=-1).flatten(1)
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        latent = self.trunk(torch.cat((visual, self.proprio(proprio)), dim=-1))
        batch = rgb.shape[0]
        return {
            "joint_mean": self.joint_mean(latent).view(batch, self.action_horizon, 7),
            "joint_log_std": self.joint_log_std(latent).view(
                batch, self.action_horizon, 7).clamp(-5.0, 0.5),
            "gripper_logits": self.gripper_logits(latent),
        }

    def loss(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        mean, log_std = output["joint_mean"], output["joint_log_std"]
        normalized = (target[..., :7] - mean) * torch.exp(-log_std)
        joint_nll = 0.5 * (normalized.square() + 2.0 * log_std).mean()
        gripper = F.binary_cross_entropy_with_logits(
            output["gripper_logits"], target[..., 7])
        return joint_nll + gripper

    @torch.no_grad()
    def sample(self, rgb: torch.Tensor, robot_history: torch.Tensor,
               joint_history: torch.Tensor, count: int, seed: int,
               temperature: float = 1.0) -> torch.Tensor:
        output = self(rgb, robot_history, joint_history)
        mean = output["joint_mean"].expand(count, -1, -1)
        std = output["joint_log_std"].exp().expand_as(mean)
        generator = torch.Generator(device=mean.device).manual_seed(seed)
        joints = mean + float(temperature) * std * torch.randn(
            mean.shape, generator=generator, device=mean.device)
        joints = torch.maximum(torch.minimum(joints, self.joint_max), self.joint_min)
        probability = output["gripper_logits"].sigmoid().expand(count, -1)
        gripper = (torch.rand(probability.shape, generator=generator,
                              device=probability.device) < probability).float()
        return torch.cat((joints, gripper[..., None]), dim=-1)


@lru_cache(maxsize=4)
def _load(checkpoint: str, device_name: str) -> FrozenBCPolicy:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
    if state.get("action_semantics") != "absolute_joint_position":
        raise ValueError("BC checkpoint does not declare absolute joint-position actions")
    model = FrozenBCPolicy(**state["model_config"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def provide(observation_history: dict[str, np.ndarray], candidate_count: int,
            action_horizon: int, seed: int, checkpoint: str,
            device: str = "cuda", temperature: str | float = 1.0) -> np.ndarray:
    """CandidateProvider entry point loaded by the RLBench collector."""
    model = _load(str(Path(checkpoint).expanduser()), device)
    if action_horizon != model.action_horizon:
        raise ValueError(
            f"collector requested horizon {action_horizon}; BC was trained for "
            f"{model.action_horizon}")
    rgb = np.asarray(observation_history["rgb_history"][-1], dtype=np.uint8)
    if rgb.shape[-1] == 3:
        rgb = np.transpose(rgb, (2, 0, 1))
    rgb_tensor = torch.from_numpy(rgb.copy()).float()[None] / 255.0
    rgb_tensor = F.interpolate(
        rgb_tensor, (model.image_size, model.image_size), mode="bilinear", align_corners=False)
    robot = torch.from_numpy(np.asarray(
        observation_history["robot_state"], dtype=np.float32))[None]
    joints = torch.from_numpy(np.asarray(
        observation_history["joint_history"], dtype=np.float32))[None]
    device_obj = next(model.parameters()).device
    actions = model.sample(rgb_tensor.to(device_obj), robot.to(device_obj),
                           joints.to(device_obj), candidate_count, seed,
                           float(temperature))
    return actions.cpu().numpy().astype(np.float32)
