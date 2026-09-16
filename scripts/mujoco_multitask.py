"""Multi-task MuJoCo factor framework.

Four tasks share one contract:
  - action  = a deterministic "direction" chunk (fixed amplitude, scalar-matched)
  - geometry = programmable target/object positions
  - outcome = success / wrong_target / drop (decided ONLY by final geometry)

Each task reproduces the amplitude confound and the paired-geometry
counterfactual, then reports action-only / geometry-only / action+geometry
AUROC. This is the multi-task generalization of mujoco_factor_task.py.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------- task base
class MuJoCoTask:
    task_name = "base"

    def _model(self):
        raise NotImplementedError

    def __init__(self, seed=0):
        import mujoco
        self.mujoco = mujoco
        self.model, self.data = self._model()
        self.rng = np.random.default_rng(seed)

    def reset(self, **kw):
        self.mujoco.mj_resetData(self.model, self.data)
        self._reset_state(**kw)
        self.mujoco.mj_forward(self.model, self.data)

    def _reset_state(self, **kw):
        raise NotImplementedError

    def set_geometry(self, geometry):
        """Write a geometry configuration into the model (target/object pos)."""
        raise NotImplementedError

    def rollout(self, direction, amplitude):
        """Run a fixed action chunk; return dict(label, success, target, ...)."""
        raise NotImplementedError

    def confound_rows(self, n):
        """Fixed geometry + fixed direction, amplitude decides the outcome."""
        raise NotImplementedError

    def interaction_rows(self, n):
        """Balanced direction x geometry, fixed amplitude."""
        raise NotImplementedError


# ---------------------------------------------------------------- task 1: seat-place
class SeatPlaceTask(MuJoCoTask):
    task_name = "seat_place"

    def _model(self):
        import mujoco
        xml = r'''
        <mujoco model="seat_place">
          <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
          <worldbody>
            <body name="payload" pos="0 0 0.08"><freejoint/>
              <geom type="cylinder" size="0.026 0.012" mass="0.03" rgba="0.9 0.2 0.1 1"/>
            </body>
            <body name="ee" mocap="true" pos="0 0 0.12">
              <geom type="sphere" size="0.012" mass="0.01" rgba="0.1 0.3 0.9 1"/>
            </body>
            <geom name="table" type="plane" size="1 1 0.01" rgba="0.25 0.25 0.25 1"/>
            <site name="seat0" pos="0.16 0 0.012" size="0.025"/>
            <site name="seat1" pos="-0.08 0.1386 0.012" size="0.025"/>
            <site name="seat2" pos="-0.08 -0.1386 0.012" size="0.025"/>
            <geom name="sg0" type="cylinder" pos="0.16 0 0.006" size="0.042 0.010"/>
            <geom name="sg1" type="cylinder" pos="-0.08 0.1386 0.006" size="0.042 0.010"/>
            <geom name="sg2" type="cylinder" pos="-0.08 -0.1386 0.006" size="0.042 0.010"/>
          </worldbody>
        </mujoco>'''
        return mujoco.MjModel.from_xml_string(xml), mujoco.MjData(mujoco.MjModel.from_xml_string(xml))

    def _reset_state(self, payload_xy=None):
        if payload_xy is None:
            payload_xy = self.rng.uniform(-.025, .025, 2)
        self.payload_xy = np.asarray(payload_xy, np.float64)
        self.data.qpos[:3] = [*self.payload_xy, .08]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.mocap_pos[0] = [*self.payload_xy, .12]
        # default seats
        self.seats = np.array([[.16, 0, .012], [-.08, .1386, .012], [-.08, -.1386, .012]], np.float64)

    def set_geometry(self, geometry):
        self.seats = np.asarray(geometry, np.float64)
        for i in range(3):
            self.model.site_pos[self.model.site(f"seat{i}").id] = [self.seats[i,0], self.seats[i,1], .012]
            self.model.geom_pos[self.model.geom(f"sg{i}").id] = [self.seats[i,0], self.seats[i,1], .006]
        self.mujoco.mj_forward(self.model, self.data)

    def rollout(self, direction, amplitude, release_step=26, horizon=30):
        self.reset(self.payload_xy)
        start = self.data.mocap_pos[0].copy()
        vec = self.seats[int(direction)] - start; vec[2] = 0
        unit = vec / max(np.linalg.norm(vec[:2]), 1e-8)
        end = start.copy(); end[:2] += unit[:2] * amplitude
        for t in range(horizon):
            alpha = min(1.0, (t+1)/max(1, release_step))
            self.data.mocap_pos[0] = start*(1-alpha) + end*alpha
            self.data.mocap_pos[0,2] = .12
            if t < release_step:
                self.data.qpos[:3] = self.data.mocap_pos[0] + np.array([0,0,-.04])
                self.data.qvel[:] = 0
            self.mujoco.mj_step(self.model, self.data)
        p = self.data.qpos[:3].copy()
        d = np.linalg.norm(self.seats[:, :2] - p[:2], axis=1)
        nearest = int(np.argmin(d))
        settled = d[nearest] < .035 and p[2] < .08
        label = "success" if nearest == 0 and settled else ("wrong_target" if settled else "drop")
        return {"label": label, "success": int(label=="success"),
                "wrong": int(label=="wrong_target"), "drop": int(label=="drop"),
                "target": 0, "direction": int(direction),
                "nearest": nearest, "nearest_distance": float(d[nearest])}

    def confound_rows(self, n):
        rows = []
        for i in range(n):
            amp = 0.16 if i % 2 == 0 else 0.10
            rows.append({"amplitude": amp, "direction": 0, "success": int(amp >= .15)})
        return rows

    def interaction_rows(self, n):
        rows = []
        for i in range(n):
            direction = i % 3
            theta = self.rng.uniform(-np.pi, np.pi); r = self.rng.uniform(.145, .175)
            angles = theta + np.deg2rad([0., 120., 240.])
            geom = np.c_[r*np.cos(angles), r*np.sin(angles), np.full(3, .012)]
            self.set_geometry(geom)
            res = self.rollout(direction=direction, amplitude=.16)
            res["target_xy"] = self.seats[0, :2].tolist()
            res["seat_positions"] = self.seats[:, :2].reshape(-1).tolist()
            rows.append(res)
        return rows


# ---------------------------------------------------------------- task 2: reach
class ReachTask(MuJoCoTask):
    task_name = "reach"

    def _model(self):
        import mujoco
        xml = r'''
        <mujoco model="reach">
          <option timestep="0.01" gravity="0 0 0" integrator="implicitfast"/>
          <worldbody>
            <body name="ee" mocap="true" pos="0 0 0.1">
              <geom type="sphere" size="0.012" rgba="0.1 0.3 0.9 1"/>
            </body>
            <site name="target" pos="0.15 0 0.1" size="0.03" rgba="0.9 0.2 0.1 1"/>
          </worldbody>
        </mujoco>'''
        m = mujoco.MjModel.from_xml_string(xml)
        return m, mujoco.MjData(m)

    def _reset_state(self, ee_xy=None):
        if ee_xy is None:
            ee_xy = self.rng.uniform(-.02, .02, 2)
        self.ee_xy = np.asarray(ee_xy, np.float64)
        self.data.mocap_pos[0] = [*self.ee_xy, .1]
        self.target_xy = np.array([.15, 0.], np.float64)

    def set_geometry(self, geometry):
        self.target_xy = np.asarray(geometry, np.float64)
        self.model.site_pos[self.model.site("target").id] = [*self.target_xy, .1]
        self.mujoco.mj_forward(self.model, self.data)

    def rollout(self, direction, amplitude, horizon=25):
        self.reset(self.ee_xy)
        start = self.data.mocap_pos[0].copy()
        # three fixed world-space directions (120 deg apart), scalar-matched
        dirs = np.array([[1.,0.,0.],[-.5,.866,0.],[-.5,-.866,0.]])
        end = start + dirs[int(direction)] * amplitude
        for t in range(horizon):
            a = min(1.0, (t+1)/horizon)
            self.data.mocap_pos[0] = start*(1-a) + end*a
            self.mujoco.mj_step(self.model, self.data)
        p = self.data.mocap_pos[0].copy()
        d = np.linalg.norm(p[:2] - self.target_xy)
        success = d < .04
        return {"label": "success" if success else "miss", "success": int(success),
                "wrong": 0, "drop": int(not success), "target": 0,
                "direction": int(direction), "nearest": 0, "nearest_distance": float(d)}

    def confound_rows(self, n):
        rows = []
        for i in range(n):
            amp = .18 if i % 2 == 0 else .08
            rows.append({"amplitude": amp, "direction": 0, "success": int(amp >= .15)})
        return rows

    def interaction_rows(self, n):
        rows = []
        for i in range(n):
            direction = i % 3
            # target placed at the endpoint of one of three directions (scalar-matched)
            dirs = np.array([[1.,0.],[-.5,.866],[-.5,-.866]])
            target_dir = self.rng.integers(0, 3)
            geom = dirs[target_dir] * .15
            self.set_geometry(geom)
            res = self.rollout(direction=direction, amplitude=.15)
            res["target_xy"] = self.target_xy.tolist()
            res["seat_positions"] = self.target_xy.tolist() * 3
            rows.append(res)
        return rows


# ---------------------------------------------------------------- audit (shared)
def audit(rows, confound_rows=None):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    directions = np.eye(3)[np.asarray([r["direction"] for r in rows])]
    X_action = np.c_[directions]
    X_geom = np.asarray([r["target_xy"] + r["seat_positions"] for r in rows])
    y = np.asarray([r["success"] for r in rows])
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); cut = len(y)//2
    tr, te = idx[:cut], idx[cut:]
    out = {"n": len(y), "positive_rate": float(y.mean())}
    for name, X in (("action_only", X_action), ("geometry_only", X_geom),
                    ("action_plus_geometry", np.c_[X_action, X_geom])):
        clf = (MLPClassifier(hidden_layer_sizes=(32,32), max_iter=2000, random_state=0)
               if name == "action_plus_geometry" else LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        out[name+"_auroc"] = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:,1]))
    if confound_rows is not None:
        xc = np.asarray([[r["amplitude"]] for r in confound_rows])
        yc = np.asarray([r["success"] for r in confound_rows])
        cc = LogisticRegression(max_iter=2000).fit(xc, yc)
        out["confound_amplitude_auroc"] = float(roc_auc_score(yc, cc.predict_proba(xc)[:,1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/diagnostics/mujoco_multitask")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--tasks", default="seat_place,reach")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    TASKS = {"seat_place": SeatPlaceTask, "reach": ReachTask}
    report = {}
    for name in args.tasks.split(","):
        name = name.strip()
        if name not in TASKS:
            continue
        task = TASKS[name](seed=2027)
        conf = task.confound_rows(args.n)
        inter = task.interaction_rows(args.n)
        report[name] = {"confound_success_rate": float(np.mean([r["success"] for r in conf])),
                        "audit": audit(inter, conf)}
        print(f"[{name}]")
        print(json.dumps(report[name], indent=2))
    (out/"report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
