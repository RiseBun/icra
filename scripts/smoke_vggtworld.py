"""Endpoint smoke for PointWorld wrapper -> spatial risk head.

By default this uses a cached Omega export so it is fast and works without
reloading the 1B-parameter front-end.  ``--rgb`` additionally reruns Omega on
the stored RGB history and therefore requires the Omega checkout/checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import (PointWorldDynamicsWrapper, RiskFrom4DBeliefModel,
                    VGGTOmegaSceneAdapter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--pointworld-checkpoint", required=True)
    parser.add_argument("--risk-checkpoint")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--omega-repo", default="~/VGGT-Omega")
    parser.add_argument("--omega-checkpoint", default="~/VGGT-Omega/checkpoints/vggt_omega_1b_512.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.sample, allow_pickle=False) as archive:
        sample = {key: np.array(archive[key], copy=True) for key in archive.files}
    if args.rgb:
        adapter = VGGTOmegaSceneAdapter({
            "repo": args.omega_repo, "checkpoint": args.omega_checkpoint,
            "device": str(device),
        })
        output = adapter.encode(sample["history_rgb"], sample["camera_intrinsics"],
                                sample["camera_extrinsics"], sample["robot_base_to_world"],
                                point_count=sample["points"].shape[1],
                                scale=float(sample.get("omega_scale", 1.0)))
        points, features = output.points[None], output.point_features[None]
        geometry_source = output.metadata
    else:
        points = torch.from_numpy(sample["points"])[None]
        features = torch.from_numpy(sample["point_features"])[None]
        geometry_source = {"geometry_source": sample.get("geometry_source", "cached")}
    state = torch.from_numpy(sample["robot_state"])[None]
    robot_flow = torch.from_numpy(sample["robot_action_flow"])[None]
    wrapper = PointWorldDynamicsWrapper(
        official_checkpoint=Path(args.pointworld_checkpoint).with_name("model-best.pt"),
        custom_checkpoint=args.pointworld_checkpoint, device=device,
    )
    future = wrapper.predict(points, features, state, robot_flow)
    risk_config = {"point_feature_dim": features.shape[-1],
                   "robot_state_dim": state.shape[-1],
                   "future_steps": future.future_point_flow.shape[2],
                   "hidden_dim": 64, "layers": 1, "heads": 4, "dropout": 0.0}
    risk = RiskFrom4DBeliefModel(**risk_config).to(device).eval()
    if args.risk_checkpoint:
        state_dict = torch.load(args.risk_checkpoint, map_location="cpu", weights_only=True)
        risk = RiskFrom4DBeliefModel(**state_dict["model_config"]).to(device).eval()
        risk.load_state_dict(state_dict["model"])
    with torch.inference_mode():
        out = risk(points.to(device), features.to(device), state.to(device),
                   robot_flow.to(device), future.future_point_flow)
    result = {
        "backend": wrapper.info.backend,
        "backend_note": wrapper.info.note,
        "geometry_source": geometry_source,
        "future_flow_shape": list(future.future_point_flow.shape),
        "risk_shapes": {"contact": list(out.task_contact_logits.shape),
                        "collision": list(out.harmful_collision_logits.shape),
                        "success": list(out.success_logits.shape)},
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
