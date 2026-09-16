"""Cache frozen Omega layer-4 tokens for ActionWorld training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.omega_latent import OmegaLatentExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--omega-repo", default="~/VGGT-Omega")
    parser.add_argument("--omega-checkpoint", default="~/VGGT-Omega/checkpoints/vggt_omega_1b_512.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--future-batch-size", type=int, default=1,
                        help="number of candidate future clips encoded per Omega call")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.future_batch_size < 1:
        raise ValueError("future batch size must be positive")
    source, destination = Path(args.input), Path(args.output)
    files = sorted(source.glob("**/*.npz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError(f"no NPZ clips found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    extractor = OmegaLatentExtractor({"repo": args.omega_repo,
                                      "checkpoint": args.omega_checkpoint,
                                      "device": args.device})
    manifest = {"source": str(source), "layer_index": extractor.config.layer_index,
                "files": [], "future_rgb_available": True,
                "future_batch_size": args.future_batch_size}
    for path in files:
        target = destination / path.relative_to(source)
        if args.skip_existing and target.is_file():
            manifest["files"].append(str(target))
            continue
        with np.load(path, allow_pickle=False) as archive:
            sample = {key: np.array(archive[key], copy=True) for key in archive.files}
        if "history_rgb" not in sample:
            raise ValueError(f"{path} has no history_rgb")
        # Latent supervision requires future RGB frames, not only simulator flow.
        has_future = "future_rgb" in sample
        manifest["future_rgb_available"] = bool(manifest["future_rgb_available"] and has_future)
        latent = extractor.encode(sample["history_rgb"])
        sample["omega_latent_tokens"] = latent.tokens[0].cpu().numpy().astype(np.float32)
        sample["omega_scene_tokens"] = latent.scene_tokens[0].cpu().numpy().astype(np.float32)
        sample["omega_patch_tokens"] = latent.patch_tokens[0].cpu().numpy().astype(np.float32)
        sample["omega_latent_layer"] = np.asarray(extractor.config.layer_index, dtype=np.int32)
        if has_future:
            future = np.asarray(sample["future_rgb"])
            if future.ndim != 5 or future.shape[2] != 3:
                raise ValueError(f"{path} future_rgb must have shape [M,H,3,H,W]")
            token_chunks, scene_chunks, patch_chunks = [], [], []
            for start in range(0, len(future), args.future_batch_size):
                future_latent = extractor.encode(future[start:start + args.future_batch_size])
                token_chunks.append(future_latent.tokens.cpu().numpy().astype(np.float32))
                scene_chunks.append(future_latent.scene_tokens.cpu().numpy().astype(np.float32))
                patch_chunks.append(future_latent.patch_tokens.cpu().numpy().astype(np.float32))
            sample["omega_future_latent_tokens"] = np.concatenate(token_chunks, axis=0)
            sample["omega_future_scene_tokens"] = np.concatenate(scene_chunks, axis=0)
            sample["omega_future_patch_tokens"] = np.concatenate(patch_chunks, axis=0)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **sample)
        manifest["files"].append(str(target))
    manifest["clips"] = len(manifest["files"])
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
