"""Export frozen VGGT-Omega scene tokens for the MuJoCo RGB dataset."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.omega_latent import OmegaLatentConfig, OmegaLatentExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    src = np.load(args.input)
    rgb = src["rgb"]
    if args.limit > 0: rgb = rgb[:args.limit]
    # Each MuJoCo sample is a static observation in this gate.  Repeat the
    # frame to satisfy Omega's video contract; temporal data comes later.
    frames = np.transpose(rgb, (0, 3, 1, 2))
    frames = np.repeat(frames[:, None], args.history, axis=1)
    extractor = OmegaLatentExtractor(OmegaLatentConfig(device=args.device,
                                                        resolution=256,
                                                        layer_index=4))
    pooled, scene, patches = [], [], []
    for start in range(0, len(frames), args.batch_size):
        batch = torch.from_numpy(frames[start:start + args.batch_size])
        out = extractor.encode(batch)
        # Keep the per-frame scene register tokens and a compact temporal mean;
        # patch tokens are intentionally omitted to avoid a multi-GB cache.
        scene.append(out.scene_tokens.mean(dim=1).half().cpu().numpy())
        pooled.append(out.tokens.mean(dim=(1, 2)).half().cpu().numpy())
        patches.append(out.patch_tokens.mean(dim=1).half().cpu().numpy())
        print(json.dumps({"done": min(start + args.batch_size, len(frames)),
                          "total": len(frames)}), flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    scene_np = np.concatenate(scene, axis=0)
    pooled_np = np.concatenate(pooled, axis=0)
    patch_np = np.concatenate(patches, axis=0)
    np.savez_compressed(output, scene_tokens=scene_np, pooled_tokens=pooled_np,
                        patch_tokens=patch_np,
                        success=src["success"][:len(frames)],
                        direction=src["direction"][:len(frames)],
                        target_xy=src["target_xy"][:len(frames)],
                        payload_xy=src["payload_xy"][:len(frames)],
                        source=np.asarray("frozen_vggt_omega"),
                        history=np.asarray(args.history, np.int32))
    print(json.dumps({"output": str(output), "scene_tokens": scene_np.shape,
                      "pooled_tokens": pooled_np.shape,
                      "patch_tokens": patch_np.shape}), flush=True)


if __name__ == "__main__": main()
