"""Amplitude vs direction on the official FACT transformer value head.

Loads CasualWorldActionTransformer only. Dummy ref latents and dummy T5.
Does not download Wan 5B and does not retrain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


EXPECTED_BYTES = 10387499812


def _load_transformer(fact_root: Path, transformer_dir: Path, device: str, dtype: torch.dtype):
    sys.path.insert(0, str(fact_root))
    from world_action_model import apply_runtime_compat
    apply_runtime_compat()
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer

    cfg = dict(CasualWorldActionTransformer.load_config(transformer_dir))
    cfg["robot_adapter_rank"] = int(cfg.get("robot_adapter_rank") or 0)
    weights = transformer_dir / "diffusion_pytorch_model.bin"
    if weights.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"incomplete weights: {weights.stat().st_size} bytes, expected {EXPECTED_BYTES}"
        )
    state = torch.load(str(weights), map_location="cpu", weights_only=False)
    model = CasualWorldActionTransformer.from_config(cfg, torch_dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.set_self_attention_implementation("sdpa")
    model.to(device=device, dtype=dtype).eval()
    return model, {"missing": len(missing), "unexpected": len(unexpected), "cfg": cfg}


def _unit_action(chunk: int, dim: int, rotate: bool) -> torch.Tensor:
    vec = torch.zeros(1, chunk, dim)
    axis = 1 if rotate else 0
    vec[..., axis] = 1.0
    return vec


def sweep(model, device: str, dtype: torch.dtype, chunk: int = 8, dim: int = 14):
    B, C, F, H, W = 1, 48, 1, 16, 16
    ref = torch.zeros(B, C, F, H, W, device=device, dtype=dtype)
    text = torch.zeros(B, 64, 4096, device=device, dtype=dtype)
    state = torch.zeros(B, 1, dim, device=device, dtype=torch.float32)
    g = torch.Generator(device="cpu").manual_seed(0)
    value_noise = torch.randn(B, 1, 1, generator=g)
    future_noise = torch.randn(B, 1, dim, generator=g)
    pred_noise = torch.randn(B, chunk, dim, generator=g)
    timestep = torch.full((B,), 500, device=device, dtype=torch.long)
    scales = [0.0, 0.5, 1.0, 2.0, 4.0]
    rows = []
    with torch.no_grad():
        for scale in scales:
            for rotate in (False, True):
                gt = (_unit_action(chunk, dim, rotate) * float(scale)).to(device)
                try:
                    out = model(
                        ref_latents=ref,
                        timestep=timestep,
                        encoder_hidden_states=text,
                        state=state.to(device),
                        action=pred_noise.to(device),
                        gt_action=gt,
                        future_state=future_noise.to(device),
                        value=value_noise.to(device),
                        action_only=True,
                        return_dict=False,
                    )
                    value_pred = out[3]
                    mag = float(value_pred.float().mean().cpu()) if value_pred is not None else float("nan")
                    err = None if value_pred is not None else "value_pred is None"
                except Exception as exc:
                    mag, err = float("nan"), f"{type(exc).__name__}: {exc}"
                rows.append({
                    "scale": scale, "rotate": rotate, "value_pred_mean": mag, "error": err,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact-root", required=True)
    ap.add_argument("--transformer-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    transformer_dir = Path(args.transformer_dir)
    weights = transformer_dir / "diffusion_pytorch_model.bin"
    result = {
        "weights": str(weights),
        "bytes": weights.stat().st_size if weights.exists() else 0,
        "expected_bytes": EXPECTED_BYTES,
        "complete": weights.exists() and weights.stat().st_size == EXPECTED_BYTES,
    }
    if not result["complete"]:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model, load_info = _load_transformer(Path(args.fact_root), transformer_dir, args.device, dtype)
    result["load"] = load_info
    result["sweep"] = sweep(model, args.device, dtype)
    ax = [r["scale"] for r in result["sweep"] if not r["rotate"] and r["error"] is None]
    ay = [r["value_pred_mean"] for r in result["sweep"] if not r["rotate"] and r["error"] is None]
    if len(ax) >= 3:
        result["scale_vs_value_pearson"] = float(np.corrcoef(ax, ay)[0, 1])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
