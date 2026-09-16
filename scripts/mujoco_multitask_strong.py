"""Multi-task strong-form factor framework.

Two (then more) tasks share the strong-form contract:
  - action = FIXED world-space direction + FIXED amplitude (scalar-matched)
  - geometry = CONTINUOUS target/object position
  - outcome decided by direction x geometry, NEVER by amplitude

confound: fixed target + fixed direction + varying amplitude -> amplitude
          moves the endpoint, which decides success via a REAL rollout.
interaction: fixed amplitude + 3 directions + continuous target -> direction
          x target-position decides success.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

DIRS = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]])
SUCCESS_RADIUS = 0.045


def _reach_model():
    import mujoco
    xml = r'''
    <mujoco model="reach">
      <option timestep="0.01" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <body name="ee" mocap="true" pos="0 0 0.1"><geom type="sphere" size="0.012" rgba="0.1 0.3 0.9 1"/></body>
        <site name="target" pos="0.15 0 0.1" size="0.03" rgba="0.9 0.2 0.1 1"/>
      </worldbody>
    </mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    return mujoco, m, mujoco.MjData(m)


def _push_model():
    import mujoco
    xml = r'''
    <mujoco model="push">
      <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
      <worldbody>
        <geom name="table" type="plane" size="1 1 0.01" rgba="0.3 0.3 0.3 1"/>
        <body name="obj" pos="0 0 0.03"><freejoint/>
          <geom type="box" size="0.03 0.03 0.03" mass="0.05" rgba="0.9 0.6 0.1 1"/>
        </body>
        <body name="ee" mocap="true" pos="0 0 0.05"><geom type="sphere" size="0.015" rgba="0.1 0.3 0.9 1"/></body>
        <site name="target" pos="0.15 0 0.03" size="0.03" rgba="0.9 0.2 0.1 1"/>
      </worldbody>
    </mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    return mujoco, m, mujoco.MjData(m)


class ReachTask:
    task_name = "reach"

    def __init__(self, seed=0):
        self.mujoco, self.model, self.data = _reach_model()
        self.rng = np.random.default_rng(seed)
        self.target_xy = np.array([0.15, 0.0])

    def set_geometry(self, target_xy):
        self.target_xy = np.asarray(target_xy, np.float64)
        self.model.site_pos[self.model.site("target").id] = [*self.target_xy, 0.1]
        self.mujoco.mj_forward(self.model, self.data)

    def rollout(self, direction, amplitude, horizon=25):
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[0] = [0.0, 0.0, 0.1]
        self.mujoco.mj_forward(self.model, self.data)
        start = self.data.mocap_pos[0].copy()
        end = start + DIRS[int(direction)] * amplitude
        for t in range(horizon):
            a = min(1.0, (t + 1) / horizon)
            self.data.mocap_pos[0] = start * (1 - a) + end * a
            self.mujoco.mj_step(self.model, self.data)
        p = self.data.mocap_pos[0].copy()
        d = float(np.linalg.norm(p[:2] - self.target_xy))
        success = d < SUCCESS_RADIUS
        return {"label": "success" if success else "miss", "success": int(success),
                "wrong": 0, "drop": int(not success), "direction": int(direction),
                "endpoint_xy": p[:2].tolist(), "target_distance": d}

    def confound_rows(self, n):
        rows = []
        self.set_geometry(DIRS[0][:2] * 0.16)
        for i in range(n):
            amp = 0.18 if i % 2 == 0 else 0.08
            res = self.rollout(direction=0, amplitude=amp)
            rows.append({"amplitude": amp, "direction": 0, "success": res["success"],
                         "target_distance": res["target_distance"]})
        return rows

    def interaction_rows(self, n):
        rows = []
        for i in range(n):
            direction = i % 3
            if (i // 3) % 2 == 0:
                target_xy = DIRS[direction][:2] * 0.16 + self.rng.uniform(-0.06, 0.06, 2)
            else:
                other = (direction + 1 + self.rng.integers(0, 2)) % 3
                target_xy = DIRS[other][:2] * 0.16 + self.rng.uniform(-0.06, 0.06, 2)
            self.set_geometry(target_xy)
            res = self.rollout(direction=direction, amplitude=0.16)
            res["target_xy"] = self.target_xy.tolist()
            rows.append(res)
        return rows


class PushTask:
    task_name = "push"

    def __init__(self, seed=0):
        self.mujoco, self.model, self.data = _push_model()
        self.rng = np.random.default_rng(seed)
        self.target_xy = np.array([0.15, 0.0])

    def set_geometry(self, target_xy):
        self.target_xy = np.asarray(target_xy, np.float64)
        self.model.site_pos[self.model.site("target").id] = [*self.target_xy, 0.03]
        self.mujoco.mj_forward(self.model, self.data)

    def rollout(self, direction, amplitude, horizon=60):
        self.mujoco.mj_resetData(self.model, self.data)
        # object starts at origin on the table
        self.data.qpos[:3] = [0.0, 0.0, 0.03]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        # ee starts BEHIND the object (opposite the push direction)
        start = -DIRS[direction] * 0.06
        self.data.mocap_pos[0] = [start[0], start[1], 0.03]
        self.mujoco.mj_forward(self.model, self.data)
        # sweep the ee through the object along DIRS[direction]
        end = DIRS[direction] * amplitude
        end_pos = np.array([end[0], end[1], 0.03])
        for t in range(horizon):
            a = min(1.0, (t + 1) / horizon)
            self.data.mocap_pos[0] = start * (1 - a) * 0 + np.array(
                [start[0] + (end[0] - start[0]) * a,
                 start[1] + (end[1] - start[1]) * a, 0.03])
            self.mujoco.mj_step(self.model, self.data)
        p = self.data.qpos[:2].copy()
        d = float(np.linalg.norm(p - self.target_xy))
        moved = float(np.linalg.norm(p)) > 0.02
        success = d < SUCCESS_RADIUS
        return {"label": "success" if success else "miss", "success": int(success),
                "wrong": 0, "drop": int(not success and not moved),
                "direction": int(direction), "endpoint_xy": p.tolist(),
                "target_distance": d, "moved": float(moved)}

    def confound_rows(self, n):
        rows = []
        self.set_geometry(DIRS[0][:2] * 0.20)  # target where amp=0.16 lands the object
        for i in range(n):
            amp = 0.16 if i % 2 == 0 else 0.08
            res = self.rollout(direction=0, amplitude=amp)
            rows.append({"amplitude": amp, "direction": 0, "success": res["success"],
                         "target_distance": res["target_distance"]})
        return rows

    def interaction_rows(self, n):
        rows = []
        for i in range(n):
            direction = i % 3
            if (i // 3) % 2 == 0:
                target_xy = DIRS[direction][:2] * 0.20 + self.rng.uniform(-0.06, 0.06, 2)
            else:
                other = (direction + 1 + self.rng.integers(0, 2)) % 3
                target_xy = DIRS[other][:2] * 0.20 + self.rng.uniform(-0.06, 0.06, 2)
            self.set_geometry(target_xy)
            res = self.rollout(direction=direction, amplitude=0.16)
            res["target_xy"] = self.target_xy.tolist()
            rows.append(res)
        return rows


def audit(rows, confound_rows=None):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    directions = np.eye(3)[np.asarray([r["direction"] for r in rows])]
    X_action = np.c_[directions]
    X_geom = np.asarray([r["target_xy"] for r in rows])
    y = np.asarray([r["success"] for r in rows])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y)); cut = len(y) // 2
    tr, te = idx[:cut], idx[cut:]
    out = {"n": len(y), "positive_rate": float(y.mean())}
    for name, X in (("action_only", X_action), ("geometry_only", X_geom),
                    ("action_plus_geometry", np.c_[X_action, X_geom])):
        clf = (MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=0)
               if name == "action_plus_geometry" else LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        out[name + "_auroc"] = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    out["marginal_gain"] = float(out["action_plus_geometry_auroc"] - out["action_only_auroc"])
    if confound_rows is not None:
        xc = np.asarray([[r["amplitude"]] for r in confound_rows])
        yc = np.asarray([r["success"] for r in confound_rows])
        cc = LogisticRegression(max_iter=2000).fit(xc, yc)
        out["confound_amplitude_auroc"] = float(roc_auc_score(yc, cc.predict_proba(xc)[:, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/diagnostics/mujoco_multitask_strong")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--tasks", default="reach,push")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    TASKS = {"reach": ReachTask, "push": PushTask}
    report = {}
    for name in args.tasks.split(","):
        name = name.strip()
        task = TASKS[name](seed=2027)
        conf = task.confound_rows(args.n)
        inter = task.interaction_rows(args.n)
        report[name] = {
            "confound_success_rate": float(np.mean([r["success"] for r in conf])),
            "audit": audit(inter, conf),
        }
        print(f"[{name}]")
        print(json.dumps(report[name], indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
