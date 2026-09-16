"""Amplitude vs direction sensitivity on the official PointWorld checkpoint.

Fixes the scene, scales or rotates robot action point-flows, reports predicted
scene-flow magnitude. Does not retrain PointWorld.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _strip_module(state):
    if not any(k.startswith("module.") for k in state):
        return state
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def inspect_checkpoint(path: Path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    contract = ckpt.get("model_contract", {})
    data = ckpt.get("data_contract", {})
    state = _strip_module(ckpt["model"])
    info = {
        "checkpoint": str(path),
        "bytes": path.stat().st_size,
        "schema": ckpt.get("checkpoint_schema_version"),
        "model_contract": contract,
        "data_contract": data,
        "n_state_keys": len(state),
        "has_robot_proj": "robot_proj.fc1.weight" in state,
    }
    if "robot_proj.fc1.weight" in state:
        info["robot_features_dim"] = int(state["robot_proj.fc1.weight"].shape[1])
    return ckpt, state, info


def make_structured_batch(data_info, scale=1.0, rotate=False, seed=0):
    """Table + cube + gripper translating along a unit vector.

    robot_flows are absolute 3D coordinates over time (PointWorld contract).
    robot_features stay appearance-like and are not scaled.
    """
    B, T, Ns, Nr = 1, 11, 256, 32
    Fr = int(data_info["robot_features_dim"])
    Ds = int(data_info["scene_features_dim"])
    rng = np.random.default_rng(seed)
    n_table, n_cube = 192, 64
    table_xy = rng.uniform(-0.20, 0.20, (n_table, 2)).astype(np.float32)
    table = np.concatenate([table_xy, np.zeros((n_table, 1), np.float32)], axis=1)
    cube_c = np.array([0.04, 0.00, 0.025], np.float32)
    cube = cube_c + rng.uniform(-0.025, 0.025, (n_cube, 3)).astype(np.float32)
    cube[:, 2] = np.abs(cube[:, 2])
    scene_coord = np.concatenate([table, cube], axis=0)[None].astype(np.float32)
    direction = np.array([0.0, 1.0, 0.0], np.float32) if rotate else np.array([1.0, 0.0, 0.0], np.float32)
    gripper0 = np.array([0.00, 0.00, 0.08], np.float32)
    local = rng.normal(0, 0.008, (Nr, 3)).astype(np.float32)
    robot_coord = np.zeros((B, T, Nr, 3), np.float32)
    for t in range(T):
        frac = t / max(T - 1, 1)
        center = gripper0 + frac * float(scale) * 0.05 * direction
        robot_coord[0, t] = center + local
    scene_feat = rng.normal(0, 0.05, (B, T, Ns, Ds)).astype(np.float32)
    robot_feat = rng.normal(0, 0.05, (B, T, Nr, Fr)).astype(np.float32)
    cube_mask = np.zeros((Ns,), dtype=bool)
    cube_mask[n_table:] = True
    batch = {
        "scene_flows": torch.from_numpy(np.repeat(scene_coord[:, None], T, axis=1)),
        "scene_features": torch.from_numpy(scene_feat),
        "scene_exists": torch.ones(B, T, Ns, dtype=torch.bool),
        "robot_flows": torch.from_numpy(robot_coord),
        "robot_features": torch.from_numpy(robot_feat),
        "robot_exists": torch.ones(B, T, Nr, dtype=torch.bool),
        "__domain__": ["droid"],
    }
    batch["scene_flows"][:, 0] = torch.from_numpy(scene_coord)
    return batch, cube_mask


def _install_flash_attn_stub():
    import types
    import torch.nn.functional as F
    if "flash_attn.flash_attn_interface" in sys.modules:
        return
    fa = types.ModuleType("flash_attn")
    fai = types.ModuleType("flash_attn.flash_attn_interface")

    def flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen=None, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
    ):
        q, k, v = qkv.unbind(dim=1)
        cu = cu_seqlens.detach().cpu().tolist()
        chunks = []
        for i in range(len(cu) - 1):
            s, e = int(cu[i]), int(cu[i + 1])
            qi = q[s:e].permute(1, 0, 2).unsqueeze(0)
            ki = k[s:e].permute(1, 0, 2).unsqueeze(0)
            vi = v[s:e].permute(1, 0, 2).unsqueeze(0)
            oi = F.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=float(dropout_p), scale=softmax_scale)
            chunks.append(oi.squeeze(0).permute(1, 0, 2))
        return torch.cat(chunks, dim=0)

    fai.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    sys.modules["flash_attn"] = fa
    sys.modules["flash_attn.flash_attn_interface"] = fai


def _stub_scene_encoder(channels: int):
    import torch.nn as nn

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.channels = channels

        def forward(self, data_dict):
            exists = data_dict["scene_exists"][:, 0]
            B, Ns = exists.shape
            return torch.zeros(B, Ns, self.channels, device=exists.device, dtype=torch.float32)

    return Stub()


def try_forward(ckpt, state, info, pointworld_root: Path, device: str):
    sys.path.insert(0, str(pointworld_root))
    _install_flash_attn_stub()
    import scene_featurizer as _sf
    _sf.SceneFeatureEncoder = lambda *a, **k: _stub_scene_encoder(int(info["model_contract"].get("predictor_dim", 128)))
    from pointworld.base import BaseModel

    args = ckpt["args"]
    args.distributed = False
    args.device = device
    args.norm_stats_path = str(pointworld_root / "stats" / "droid")
    if not hasattr(args, "predictor_dim"):
        args.predictor_dim = int(info["model_contract"].get("predictor_dim", 128))
    data_info = {
        "robot_features_dim": info["robot_features_dim"],
        "scene_features_dim": int(state["scene_feature_encoder.proj.weight"].shape[1])
        if "scene_feature_encoder.proj.weight" in state
        else 32,
    }
    # find any weight that reveals scene dim
    for k, v in state.items():
        if "scene" in k and "weight" in k and getattr(v, "ndim", 0) == 2:
            data_info["scene_features_dim"] = int(v.shape[1])
            info["scene_weight_key"] = k
            break
    model = BaseModel(args, data_info, rank=0, cpu_pg=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    info["load_missing"] = len(missing)
    info["load_unexpected"] = len(unexpected)
    info["scene_features_dim"] = data_info["scene_features_dim"]

    scales = [0.0, 0.5, 1.0, 2.0, 4.0]
    rows = []
    with torch.no_grad():
        for scale in scales:
            for rotate in (False, True):
                batch, cube_mask = make_structured_batch(data_info, scale=scale, rotate=rotate)
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                try:
                    out = model(batch, training=False)
                    pred = out["scene_relative"]
                    mag = pred.pow(2).sum(-1).sqrt()
                    while mag.ndim > 1:
                        mag = mag.mean(0)
                    cube_t = torch.from_numpy(cube_mask).to(mag.device)
                    row = {
                        "scale": scale,
                        "rotate": rotate,
                        "pred_flow_l2": float(mag.mean().cpu()),
                        "cube_flow_l2": float(mag[cube_t].mean().cpu()),
                        "table_flow_l2": float(mag[~cube_t].mean().cpu()),
                        "error": None,
                    }
                except Exception as exc:
                    row = {
                        "scale": scale, "rotate": rotate,
                        "pred_flow_l2": float("nan"),
                        "cube_flow_l2": float("nan"),
                        "table_flow_l2": float("nan"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
    info["probe"] = (
        "structured table+cube; robot_flows=absolute gripper trajectory; "
        "robot_features unscaled; scene encoder stubbed to zeros"
    )
    ax = [r["scale"] for r in rows if not r["rotate"] and r.get("pred_flow_l2") == r.get("pred_flow_l2")]
    ay = [r["pred_flow_l2"] for r in rows if not r["rotate"] and r.get("pred_flow_l2") == r.get("pred_flow_l2")]
    if len(ax) >= 3:
        info["scale_vs_flow_pearson"] = float(np.corrcoef(ax, ay)[0, 1])
        info["scale0_flow"] = ay[0]
        info["scale4_flow"] = ay[-1]
        info["flow_ratio_4_over_0"] = float(ay[-1] / max(ay[0], 1e-12))
    return info, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pointworld-root", default="third_party/PointWorld")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--forward", action="store_true")
    args = ap.parse_args()
    ckpt, state, info = inspect_checkpoint(Path(args.checkpoint))
    result = {"inspect": info}
    if args.forward:
        try:
            info2, rows = try_forward(
                ckpt, state, info, Path(args.pointworld_root).resolve(), args.device)
            result["inspect"] = info2
            result["amplitude_sweep"] = rows
        except Exception as exc:
            result["forward_error"] = f"{type(exc).__name__}: {exc}"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
