"""Run live Omega latent extraction and action-conditioned transition smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import OmegaActionWorld, OmegaLatentExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--omega-repo", default="~/VGGT-Omega")
    parser.add_argument("--omega-checkpoint", default="~/VGGT-Omega/checkpoints/vggt_omega_1b_512.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    with np.load(args.sample, allow_pickle=False) as archive:
        rgb = np.array(archive["history_rgb"], copy=True)
        state = torch.from_numpy(np.array(archive["robot_state"], copy=True)).float()[None]
        robot_flow = torch.from_numpy(np.array(archive["robot_action_flow"], copy=True)).float()[None]
    extractor = OmegaLatentExtractor({"repo": args.omega_repo,
                                      "checkpoint": args.omega_checkpoint,
                                      "device": device})
    latent = extractor.encode(rgb)
    model = OmegaActionWorld(latent_dim=latent.tokens.shape[-1],
                             robot_state_dim=state.shape[-1], hidden_dim=256,
                             future_steps=8, layers=2, heads=8).to(device).eval()
    with torch.inference_mode():
        output = model(latent.tokens, state.to(device), robot_flow.to(device))
    print(json.dumps({
        "latent": latent.metadata,
        "tokens": list(latent.tokens.shape),
        "future_tokens": list(output.future_tokens.shape),
        "latent_log_variance": list(output.latent_log_variance.shape),
        "frozen_omega": True,
        "trainable_transition_parameters": sum(p.numel() for p in model.parameters()),
    }, indent=2))


if __name__ == "__main__":
    main()
