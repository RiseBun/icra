"""Fit temperature scaling and one-sided conformal risk corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_paper_metrics import load_model
from scripts.train_dataset import ClipDataset


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros((), requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / log_t.exp().clamp(0.05, 20.0), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).detach())


def conformal_quantile(residual: np.ndarray, alpha: float) -> float:
    n = len(residual)
    rank = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))
    return float(np.sort(residual)[rank - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ClipDataset(args.data)
    model = load_model(args.checkpoint, device)
    success_logits, collision_logits, success, collision = [], [], [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in ds[i].items()}
            out = model(item["points"], item["actions"], item["robot_state"], item["features"])
            success_logits.append(out.success_logits.flatten())
            collision_logits.append(out.collision_logits.flatten())
            success.append(item["success"].flatten())
            collision.append(item["collision"].flatten())
    sl, cl = torch.cat(success_logits), torch.cat(collision_logits)
    sy, cy = torch.cat(success), torch.cat(collision)
    st, ct = fit_temperature(sl, sy), fit_temperature(cl, cy)
    sp = torch.sigmoid(sl / st).cpu().numpy()
    cp = torch.sigmoid(cl / ct).cpu().numpy()
    failure = 1.0 - sy.cpu().numpy()
    collision_np = cy.cpu().numpy()
    result = {
        "alpha": args.alpha,
        "samples": int(sy.numel()),
        "success_temperature": st,
        "collision_temperature": ct,
        "failure_residual_quantile": conformal_quantile(failure - (1.0 - sp), args.alpha),
        "collision_residual_quantile": conformal_quantile(collision_np - cp, args.alpha),
        "calibration_data": str(args.data),
        "checkpoint": str(args.checkpoint),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
