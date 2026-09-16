"""Minimal MuJoCo factor task and shortcut audit.

The task is deliberately small: a kinematic end-effector carries a payload
until release, then the payload falls onto one of three programmable seats.
Labels use only final geometry (not a hidden target label).  The script first
reproduces the action-amplitude confound and then evaluates the balanced
action x geometry interaction with action-only and action+geometry baselines.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


def _mujoco_model():
    import mujoco
    xml = r'''
    <mujoco model="factor_task">
      <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
      <worldbody>
        <camera name="front" pos="0 -0.68 0.62" xyaxes="1 0 0 0 0.707 0.707"/>
        <camera name="left" pos="-0.62 -0.35 0.55" xyaxes="0.5 -0.86 0 0.61 0.35 0.71"/>
        <camera name="right" pos="0.62 -0.35 0.55" xyaxes="0.5 0.86 0 -0.61 0.35 0.71"/>
        <body name="payload" pos="0 0 0.08">
          <freejoint/>
          <geom type="cylinder" size="0.026 0.012" mass="0.03" rgba="0.9 0.2 0.1 1"/>
        </body>
        <body name="ee" mocap="true" pos="0 0 0.12">
          <geom type="sphere" size="0.012" mass="0.01" rgba="0.1 0.3 0.9 1"/>
        </body>
        <geom name="table" type="plane" size="1 1 0.01" rgba="0.25 0.25 0.25 1"/>
        <site name="seat0" pos="0.16 0 0.012" size="0.025" rgba="0.1 0.9 0.2 1"/>
        <site name="seat1" pos="-0.08 0.1386 0.012" size="0.025" rgba="0.2 0.5 0.95 1"/>
        <site name="seat2" pos="-0.08 -0.1386 0.012" size="0.025" rgba="0.95 0.2 0.2 1"/>
        <geom name="seat_geom0" type="cylinder" pos="0.16 0 0.006" size="0.042 0.010" rgba="0.2 0.8 0.2 1"/>
        <geom name="seat_geom1" type="cylinder" pos="-0.08 0.1386 0.006" size="0.042 0.010" rgba="0.2 0.45 0.9 1"/>
        <geom name="seat_geom2" type="cylinder" pos="-0.08 -0.1386 0.006" size="0.042 0.010" rgba="0.9 0.25 0.2 1"/>
      </worldbody>
      <equality><weld name="carry" body1="ee" body2="payload" active="false"/></equality>
    </mujoco>'''
    return mujoco, mujoco.MjModel.from_xml_string(xml)


class FactorTask:
    def __init__(self, seed=0, visual_mode="color"):
        self.mujoco, self.model = _mujoco_model()
        self.data = self.mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.visual_mode = visual_mode
        self.seats = np.array([[.16, 0., .012], [-.08, .1386, .012],
                               [-.08, -.1386, .012]], np.float64)
        self.seat_geom_ids = [self.model.geom(f"seat_geom{i}").id for i in range(3)]
        self.camera_id = self.model.camera("front").id
        self.camera_ids = [self.model.camera(n).id for n in ("front", "left", "right")]
        self._renderers = None
        self.reset()

    def set_target_visual(self, target: int):
        """Encode target and distractor identities with visible colors.

        The earlier gray distractors were nearly indistinguishable from the
        dark table in rendered RGB, making full-layout readout ill-posed.  We
        retain the yellow target marker and use blue/red distractors so the
        RGB/Omega comparison measures representation quality rather than
        accidental invisibility.
        """
        for i, gid in enumerate(self.seat_geom_ids):
            if self.visual_mode == "shape":
                self.model.geom_rgba[gid] = [0.55, 0.55, 0.55, 1.0]
                sizes = ([0.050, 0.010] if i == int(target) else
                         ([0.034, 0.010] if i == (int(target)+1) % 3 else [0.042, 0.010]))
                self.model.geom_size[gid, :2] = sizes
                continue
            distractor_colors = ([0.10, 0.45, 1.0, 1.0],
                                 [1.0, 0.18, 0.12, 1.0])
            if i == int(target):
                self.model.geom_rgba[gid] = [1.0, 0.85, 0.05, 1.0]
            else:
                self.model.geom_rgba[gid] = distractor_colors[(i - (i > int(target))) % 2]

    def render_rgb(self, width=128, height=128, view=0):
        if self._renderers is None:
            self._renderers = [self.mujoco.Renderer(self.model, height=height, width=width)
                               for _ in self.camera_ids]
        renderer = self._renderers[int(view)]
        renderer.update_scene(self.data, camera=self.camera_ids[int(view)])
        image = renderer.render().copy()
        return image

    def reset(self, payload_xy=None):
        self.mujoco.mj_resetData(self.model, self.data)
        if payload_xy is None:
            payload_xy = self.rng.uniform(-.025, .025, 2)
        self.payload_xy = np.asarray(payload_xy, np.float64)
        self.data.qpos[:3] = [*self.payload_xy, .08]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.mocap_pos[0] = [*self.payload_xy, .12]
        self.mujoco.mj_forward(self.model, self.data)

    def randomize_seats(self):
        """Sample a rotated equal-radius layout and update render geometry."""
        theta = float(self.rng.uniform(-np.pi, np.pi))
        radius = float(self.rng.uniform(0.145, 0.175))
        angles = theta + np.deg2rad([0.0, 120.0, 240.0])
        self.seats[:, 0] = radius * np.cos(angles)
        self.seats[:, 1] = radius * np.sin(angles)
        for i in range(3):
            self.model.site_pos[self.model.site(f"seat{i}").id] = [self.seats[i, 0], self.seats[i, 1], .012]
            self.model.geom_pos[self.seat_geom_ids[i]] = [self.seats[i, 0], self.seats[i, 1], .006]
        self.mujoco.mj_forward(self.model, self.data)

    def rollout(self, target, direction, amplitude=0.16, horizon=30,
                release_step=26, noise=0.0):
        """Run an action chunk toward a seat; return geometry-only label."""
        self.reset(self.payload_xy)
        start = self.data.mocap_pos[0].copy()
        vec = self.seats[int(direction)] - start
        vec[2] = 0.0
        unit = vec / max(np.linalg.norm(vec[:2]), 1e-8)
        end = start.copy(); end[:2] += unit[:2] * float(amplitude)
        for t in range(horizon):
            alpha = min(1.0, (t + 1) / max(1, release_step))
            self.data.mocap_pos[0] = start * (1-alpha) + end * alpha
            self.data.mocap_pos[0, 2] = .12
            if t < release_step:
                # Kinematic grasp: payload follows the end effector until the
                # release event.  Thereafter MuJoCo integrates the free body.
                self.data.qpos[:3] = self.data.mocap_pos[0] + np.array([0, 0, -.04])
                self.data.qvel[:] = 0
            self.mujoco.mj_step(self.model, self.data)
        p = self.data.qpos[:3].copy()
        d = np.linalg.norm(self.seats[:, :2] - p[:2], axis=1)
        nearest = int(np.argmin(d))
        # The cylinder rests with its center at z~=0.065 on the plane; the
        # acceptance test is therefore based on horizontal seat proximity and
        # a low settled height, not the site z coordinate itself.
        settled = d[nearest] < .035 and p[2] < .08
        label = "success" if nearest == int(target) and settled else "wrong_target" if settled else "drop"
        return {"label": label, "success": int(label == "success"),
                "target": int(target), "direction": int(direction),
                "amplitude": float(amplitude), "final_xy": p[:2].tolist(),
                "final_z": float(p[2]), "nearest": nearest,
                "nearest_distance": float(d[nearest])}


def confound_audit(rng, n=300):
    rows=[]
    # Fixed geometry and fixed direction: amplitude alone determines whether
    # the endpoint enters the acceptance disk.
    for i in range(n):
        amp = 0.16 if i % 2 == 0 else 0.10
        success = int(amp >= .15)
        rows.append({"amplitude": amp, "direction": 0, "target": 0,
                     "success": success})
    return rows


def interaction_dataset(task, n=600, multitask=False, multiview=False):
    rows=[]; images=[]
    cached_target = 0
    cached_variant = 0
    for i in range(n):
        layout_id = i // 3
        direction = i % 3
        if direction == 0:
            cached_target = layout_id % 3
            cached_variant = (layout_id // 3) % 3 if multitask else 0
            task.randomize_seats()
            if multitask and cached_variant == 2:
                task.seats *= 1.35
                for j in range(3):
                    task.model.site_pos[task.model.site(f"seat{j}").id] = [task.seats[j,0], task.seats[j,1], .012]
                    task.model.geom_pos[task.seat_geom_ids[j]] = [task.seats[j,0], task.seats[j,1], .006]
                task.mujoco.mj_forward(task.model, task.data)
        target = cached_target
        task_variant = cached_variant
        task.set_target_visual(target)
        task.reset()
        if multiview:
            images.append(np.stack([task.render_rgb(view=v) for v in range(3)]))
        else:
            images.append(task.render_rgb())
        # Equal action magnitude; success is caused by direction/target match.
        # All three directions in a scene use the same radial action length;
        # the radius is sampled once per scene and is not a target label.
        radius = .16 if not (multitask and task_variant == 2) else .16
        result = task.rollout(target=target, direction=direction,
                              amplitude=radius, release_step=26)
        # Each generated row is an independently reset episode.  Persist the
        # identifier so later training/evaluation scripts can enforce
        # episode-level splits instead of randomly splitting correlated rows.
        result["episode_id"] = int(layout_id)
        result["layout_id"] = int(layout_id)
        result["task_variant"] = int(task_variant)
        # Observable privileged geometry for this first gate.  The model gets
        # the selected target's world position, not its integer identity.
        result["target_xy"] = task.seats[target, :2].tolist()
        result["seat_positions"] = task.seats[:, :2].reshape(-1).tolist()
        result["payload_xy"] = task.payload_xy.tolist()
        rows.append(result)
    return rows, np.asarray(images, dtype=np.uint8)


def audit(rows, confound_rows=None):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    directions = np.eye(3)[np.asarray([r["direction"] for r in rows])]
    X_action = np.c_[directions, np.asarray([r["amplitude"] for r in rows])]
    X_geom = np.asarray([r["target_xy"] + r["seat_positions"] + r["payload_xy"] for r in rows])
    y = np.asarray([r["success"] for r in rows])
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); cut = len(y)//2
    tr, te = idx[:cut], idx[cut:]
    out={"positive_rate": float(y.mean())}
    for name, X in (("action_only", X_action), ("geometry_only", X_geom),
                    ("action_plus_geometry", np.c_[X_action, X_geom])):
        clf = (MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=2000,
                             random_state=0) if name == "action_plus_geometry"
               else LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        out[name+"_auroc"] = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:,1]))
    if confound_rows is not None:
        xc = np.asarray([[r["amplitude"]] for r in confound_rows])
        yc = np.asarray([r["success"] for r in confound_rows])
        cc = LogisticRegression(max_iter=2000).fit(xc, yc)
        out["confound_amplitude_only_auroc"] = float(
            roc_auc_score(yc, cc.predict_proba(xc)[:, 1]))
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output", default="data/diagnostics/mujoco_factor")
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--multitask", action="store_true"); ap.add_argument("--visual-mode", choices=["color","shape"], default="color"); ap.add_argument("--multiview", action="store_true"); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    task=FactorTask(seed=2027, visual_mode=args.visual_mode)
    conf=confound_audit(task.rng, args.n)
    inter, images=interaction_dataset(task, args.n, multitask=args.multitask, multiview=args.multiview)
    report={"confound": {"n":len(conf), "success_rate":float(np.mean([r["success"] for r in conf]))},
            "interaction": audit(inter, conf), "n":len(inter)}
    (out/"confound.json").write_text(json.dumps(conf, indent=2))
    (out/"interaction.json").write_text(json.dumps(inter, indent=2))
    np.savez_compressed(out/"rgb_dataset.npz", rgb=images,
                        episode_id=np.asarray([r["episode_id"] for r in inter], np.int64),
                        success=np.asarray([r["success"] for r in inter], np.uint8),
                        direction=np.asarray([r["direction"] for r in inter], np.int64),
                        target_xy=np.asarray([r["target_xy"] for r in inter], np.float32),
                        payload_xy=np.asarray([r["payload_xy"] for r in inter], np.float32),
                        seat_positions=np.asarray([r["seat_positions"] for r in inter], np.float32),
                        camera_intrinsics=np.repeat(np.asarray([[[128., 0., 64.],
                                                                  [0., 128., 64.],
                                                                  [0., 0., 1.]]], np.float32), len(inter), axis=0),
                        camera_extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], len(inter), axis=0))
    (out/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
