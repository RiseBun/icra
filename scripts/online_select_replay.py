"""Pilot online candidate selection followed by real RLBench replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_rlbench_multiview_perturbations import TASKS, _environment_collision, _expert_actions
from export_omega_rlbench_clips import transform, sample_at_uv, umeyama_similarity


def camera_config(size):
    from rlbench.observation_config import CameraConfig, ObservationConfig
    on = CameraConfig(rgb=True, depth=True, point_cloud=True, mask=False,
                      image_size=(size, size), depth_in_meters=True)
    off = CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False)
    return ObservationConfig(
        front_camera=on, left_shoulder_camera=off, right_shoulder_camera=off,
        overhead_camera=off, wrist_camera=off, joint_positions=True,
        joint_velocities=True, joint_forces=True, gripper_open=True, gripper_pose=True,
    )


def load_model(checkpoint, state_dim, device):
    from models.action4d import ActionConditioned4DModel
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = state["model_config"]
    model = ActionConditioned4DModel(
        robot_state_dim=int(cfg["robot_state_dim"]), action_dim=int(cfg["action_dim"]),
        future_steps=int(cfg["future_steps"]), hidden_dim=int(cfg["hidden_dim"]),
        layers=int(cfg["layers"]), heads=int(cfg["heads"]),
        point_feature_dim=int(cfg["point_feature_dim"]),
        relative_actions=bool(cfg.get("relative_actions", False)),
    ).to(device)
    model.load_state_dict(state["model"]); return model.eval()


def omega_points(observations, repo, checkpoint, device, resolution=256, points=256):
    """Run frozen Omega on the live history and calibrate to simulator world points."""
    import tempfile
    from PIL import Image
    import sys
    sys.path.insert(0, str(Path(repo).expanduser()))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, obs in enumerate(observations[:8]):
            path = Path(tmp) / f"frame_{i:04d}.png"
            Image.fromarray(np.asarray(obs.front_rgb, dtype=np.uint8)).save(path)
            paths.append(str(path))
        omega = VGGTOmega().to(device).eval()
        state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
        omega.load_state_dict(state)
        with torch.inference_mode():
            images = load_and_preprocess_images(paths, image_resolution=resolution).to(device)
            pred = omega(images)
            extrinsics, intrinsics = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
            depth = pred["depth"][0, ..., 0]
            k, h, w = depth.shape
            yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            flat = torch.stack((xx.flatten(), yy.flatten()), -1)
            ids = torch.linspace(0, flat.shape[0] - 1, points, device=device).long()
            uv = flat[ids].float()
            predicted = []
            for frame in range(k):
                z = depth[frame].flatten()[ids].clamp_min(1e-4)
                fx, fy = intrinsics[0, frame, 0, 0], intrinsics[0, frame, 1, 1]
                cx, cy = intrinsics[0, frame, 0, 2], intrinsics[0, frame, 1, 2]
                predicted.append(torch.stack(((uv[:, 0] - cx) * z / fx,
                                              (uv[:, 1] - cy) * z / fy, z), -1))
            predicted = torch.stack(predicted).cpu().numpy()
            pred_extr = extrinsics[0].cpu().numpy()
        pred_h = np.concatenate((pred_extr,
                                 np.broadcast_to(np.array([0, 0, 0, 1], np.float32), (k, 1, 4))), axis=1)
        ref = pred_h[0]
        pred_ref = np.stack([transform(predicted[i], ref @ np.linalg.inv(pred_h[i])) for i in range(k)])
        gt_cloud = np.stack([np.asarray(obs.front_point_cloud, dtype=np.float32) for obs in observations[:8]])
        gt_extr = np.asarray(observations[0].misc["front_camera_extrinsics"], dtype=np.float32)
        gt_ref = transform(gt_cloud, np.linalg.inv(gt_extr))
        gt_uv = uv.detach().cpu().numpy()
        gt_samples = sample_at_uv(gt_ref, gt_uv, (h, w))
        calibration, _ = umeyama_similarity(pred_ref[0], gt_samples[0])
        return transform(pred_ref, calibration).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="open_drawer")
    ap.add_argument("--checkpoint", default="results/action4d_clean4_full_perm.pt")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--output", default="results/online_select_replay.jsonl")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-success", type=float, default=0.90)
    ap.add_argument("--history", type=int, default=8,
                    help="number of observed demo frames used by the model")
    ap.add_argument("--paired", action="store_true",
                    help="replay fixed/random/risk/oracle candidates from the same demo")
    ap.add_argument("--omega-repo", default="~/VGGT-Omega")
    ap.add_argument("--omega-checkpoint", default="~/VGGT-Omega/checkpoints/vggt_omega_1b_512.pt")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench import tasks as task_module

    env = Environment(MoveArmThenGripper(JointPosition(absolute_mode=True), Discrete()),
                      obs_config=camera_config(args.image_size), headless=True)
    env.launch(); task = env.get_task(getattr(task_module, TASKS[args.task]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w") as stream:
            for episode in range(args.episodes):
                task.sample_variation()
                demo = task.get_demos(1, live_demos=True, max_attempts=20)[0]
                observations = list(demo)
                expert = _expert_actions(demo)
                candidates = [np.asarray(expert, dtype=np.float32)]
                noise = np.asarray(expert, dtype=np.float32).copy()
                noise[:, :7] += rng.normal(0.0, 0.03, noise[:, :7].shape).astype(np.float32)
                candidates.append(noise)
                delay = np.asarray(expert, dtype=np.float32).copy()
                if len(delay) > 1:
                    delay[1:, 7] = delay[:-1, 7]
                candidates.append(delay)
                length = 128
                action_seq = np.zeros((3, length, 8), dtype=np.float32)
                for i, candidate in enumerate(candidates):
                    action_seq[i, :min(length, len(candidate))] = candidate[:length]
                history = min(args.history, len(observations), len(expert) + 1)
                points = omega_points(observations[:history], args.omega_repo,
                                      args.omega_checkpoint, device)
                robot_state = np.asarray(observations[history - 1].get_low_dim_data(), dtype=np.float32)
                if model is None:
                    model = load_model(args.checkpoint, len(robot_state), device)
                batch = {
                    "points": torch.from_numpy(points)[None].to(device),
                    "actions": torch.from_numpy(action_seq.reshape(3, -1))[None].to(device),
                    "robot_state": torch.from_numpy(robot_state)[None].to(device),
                    "features": torch.ones((1, points.shape[0], points.shape[1], 1), device=device),
                }
                with torch.inference_mode():
                    prediction = model(**batch)
                    success_prob = torch.sigmoid(prediction.success_logits[0]).cpu().numpy()
                    selected = int(success_prob.argmax())
                    if float(success_prob[selected]) < args.min_success:
                        selected = 0
                # The model is conditioned on observation history ending at
                # observations[history - 1].  RLBench stores the executed
                # action in the observation after it was applied, so the
                # replay warm-up uses history-1 recorded actions.
                # applying a candidate.  The old implementation replayed the
                # full candidate from frame 0, making online scores and
                # execution states incomparable.
                def replay(candidate):
                    _, replay_obs = task.reset_to_demo(demo)
                    for warmup in expert[:history - 1]:
                        try:
                            replay_obs, _, warmup_done = task.step(warmup)
                        except Exception:
                            return False, 1, 0
                        if warmup_done:
                            return True, 0, history - 1
                    succeeded_local = False
                    collisions_local = []
                    frames_local = 0
                    for action in candidate[history - 1:]:
                        try:
                            replay_obs, reward, terminate = task.step(action)
                        except Exception:
                            break
                        collisions_local.append(_environment_collision(task, args.task))
                        succeeded_local = succeeded_local or reward > 0.5
                        frames_local += 1
                        if terminate:
                            break
                    return succeeded_local, int(sum(collisions_local)), frames_local

                results = [replay(candidate) for candidate in candidates]
                random_selected = int(rng.integers(len(candidates)))
                oracle = int(np.argmin([int(not ok) + collision for ok, collision, _ in results]))
                if args.paired:
                    selected_result = results[selected]
                    record = {"episode": episode, "selected": selected,
                              "success_prob": success_prob.tolist(), "success": int(selected_result[0]),
                              "collisions": int(selected_result[1]), "frames": len(observations),
                              "history": history, "warmup": history - 1,
                              "paired": {"fixed": {"candidate": 0, "success": int(results[0][0]), "collisions": int(results[0][1])},
                                         "random": {"candidate": random_selected, "success": int(results[random_selected][0]), "collisions": int(results[random_selected][1])},
                                         "risk": {"candidate": selected, "success": int(results[selected][0]), "collisions": int(results[selected][1])},
                                         "oracle": {"candidate": oracle, "success": int(results[oracle][0]), "collisions": int(results[oracle][1])}}}
                else:
                    selected_result = results[selected]
                    record = {"episode": episode, "selected": selected,
                              "success_prob": success_prob.tolist(), "success": int(selected_result[0]),
                              "collisions": int(selected_result[1]), "frames": len(observations),
                              "history": history, "warmup": history - 1}
                stream.write(json.dumps(record) + "\n"); stream.flush(); print(record, flush=True)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
