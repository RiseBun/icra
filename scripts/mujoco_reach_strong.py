"""Strong-form reach factor task.

The "strong form" fixes two prior shortcuts:
  1. The action is a FIXED world-space direction (not "toward seat[k]").
  2. The confound is REAL: amplitude moves the endpoint, which either reaches a
     FIXED target or not (via rollouts), not a hand-written label.

confound:   fixed target + fixed direction, varying amplitude -> amplitude
            encodes outcome through the endpoint position.
interaction: fixed amplitude + 3 fixed directions, CONTINUOUS random target
            position -> outcome decided by direction x target-position.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def _make_model():
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
    return mujoco, m, mujoco.MjData(m)


# three fixed world directions, 120 deg apart (scalar-matched in magnitude)
DIRS = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]])
SUCCESS_RADIUS = 0.045


class ReachStrongTask:
    def __init__(self, seed=0):
        self.mujoco, self.model, self.data = _make_model()
        self.rng = np.random.default_rng(seed)
        self.target_xy = np.array([0.15, 0.0])

    def set_geometry(self, target_xy):
        self.target_xy = np.asarray(target_xy, np.float64)
        self.model.site_pos[self.model.site("target").id] = [*self.target_xy, 0.1]
        self.mujoco.mj_forward(self.model, self.data)

    def reset(self, ee_xy=None):
        self.mujoco.mj_resetData(self.model, self.data)
        if ee_xy is None:
            ee_xy = self.rng.uniform(-0.02, 0.02, 2)
        self.ee_xy = np.asarray(ee_xy, np.float64)
        self.data.mocap_pos[0] = [*self.ee_xy, 0.1]
        self.mujoco.mj_forward(self.model, self.data)

    def rollout(self, direction, amplitude, horizon=25):
        self.reset(getattr(self, "ee_xy", None))
        start = self.data.mocap_pos[0].copy()
        end = start + DIRS[int(direction)] * amplitude
        for t in range(horizon):
            a = min(1.0, (t + 1) / horizon)
            self.data.mocap_pos[0] = start * (1 - a) + end * a
            self.mujoco.mj_step(self.model, self.data)
        p = self.data.mocap_pos[0].copy()
        d = float(np.linalg.norm(p[:2] - self.target_xy))
        success = d < SUCCESS_RADIUS
        return {"label": "success" if success else "miss",
                "success": int(success), "wrong": 0, "drop": int(not success),
                "direction": int(direction), "endpoint_xy": p[:2].tolist(),
                "target_distance": d}

    def confound_rows(self, n):
        """fixed target + fixed direction, varying amplitude via real rollouts."""
        rows = []
        # target fixed at DIRS[0]*0.16; amplitude either overshoots (reaches)
        # or undershoots (misses). Success is decided by the rollout, not a label.
        self.set_geometry(DIRS[0][:2] * 0.16)
        self.ee_xy = np.array([0.0, 0.0])  # fixed initial position
        for i in range(n):
            amp = 0.18 if i % 2 == 0 else 0.08
            res = self.rollout(direction=0, amplitude=amp)
            rows.append({"amplitude": amp, "direction": 0,
                         "success": res["success"],
                         "target_distance": res["target_distance"]})
        return rows

    def interaction_rows(self, n):
        """fixed amplitude + 3 directions, continuous random target position.

        Target positions are CONTINUOUS (sampled in near/far regions around the
        three direction endpoints), but sampled so success/failure stay
        balanced. The outcome is decided by direction x target-position, not by
        a discrete index.
        """
        rows = []
        self.ee_xy = np.array([0.0, 0.0])  # fixed initial position
        for i in range(n):
            direction = i % 3
            # near: target close to this direction's endpoint (-> success)
            # far : target close to another direction's endpoint (-> miss)
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
        # distribution of target_distance by amplitude (real confound, not label)
        lo = np.mean([r["target_distance"] for r in confound_rows if r["amplitude"] < 0.1])
        hi = np.mean([r["target_distance"] for r in confound_rows if r["amplitude"] > 0.1])
        out["confound_td_small_amp"] = float(lo)
        out["confound_td_large_amp"] = float(hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/diagnostics/mujoco_reach_strong")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    task = ReachStrongTask(seed=2027)
    conf = task.confound_rows(args.n)
    inter = task.interaction_rows(args.n)
    report = {
        "confound": {
            "n": len(conf),
            "success_rate": float(np.mean([r["success"] for r in conf])),
            "small_amp_success": float(np.mean([r["success"] for r in conf if r["amplitude"] < 0.1])),
            "large_amp_success": float(np.mean([r["success"] for r in conf if r["amplitude"] > 0.1])),
        },
        "interaction": audit(inter, conf),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
