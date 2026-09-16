"""Multi-task fixed-direction x continuous-geometry factor experiment.

The experiment deliberately separates action and geometry: the action is one
of three fixed world directions with the same magnitude, while the target and
two distractors are continuously randomized.  Labels are generated from a
MuJoCo rollout endpoint and are success / wrong_target / drop.  A second
experiment trains the same predictor either on an amplitude-confounded
perturbation set or on the paired action x geometry set, then evaluates both
on the same held-out paired test set.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

DIRS = np.array([[1., 0.], [-.5, .8660254], [-.5, -.8660254]], np.float64)
R = .045

def make_world():
    import mujoco
    xml = r'''<mujoco model="strong_multi">
      <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
      <worldbody>
        <body name="payload" pos="0 0 .07"><freejoint/><geom type="sphere" size=".018" mass=".03" rgba=".9 .2 .1 1"/></body>
        <body name="ee" mocap="true" pos="0 0 .12"><geom type="sphere" size=".012" rgba=".1 .3 .9 1"/></body>
        <geom name="table" type="plane" size="1 1 .01"/>
      </worldbody><equality><weld name="carry" body1="ee" body2="payload" active="false"/></equality>
    </mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    return mujoco, m, mujoco.MjData(m)

def rollout(mujoco, model, data, task, direction, amplitude, payload_xy, return_action=False):
    mujoco.mj_resetData(model, data)
    start = np.array([payload_xy[0], payload_xy[1], .12], np.float64)
    data.mocap_pos[0] = start
    data.qpos[:3] = [payload_xy[0], payload_xy[1], .07]
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    end = start.copy(); end[:2] += DIRS[direction] * amplitude
    horizon = 30
    prev = start.copy()
    actions = []
    for t in range(horizon):
        a = min(1., (t + 1) / 26.)
        pos = start * (1-a) + end * a
        actions.append((pos - prev)[:2].tolist())
        data.mocap_pos[0] = pos
        if task in ("push", "pick_place", "insert") and t < 24:
            data.qpos[:3] = data.mocap_pos[0] + np.array([0, 0, -.05])
            data.qvel[:] = 0
        mujoco.mj_step(model, data)
        prev = pos.copy()
    if task == "insert":
        for _ in range(12):
            data.mocap_pos[0] = data.mocap_pos[0] + np.array([0, 0, -.006])
            data.qpos[:3] = data.mocap_pos[0] + np.array([0, 0, -.05])
            data.qvel[:] = 0
            mujoco.mj_step(model, data)
    p = data.qpos[:3].copy() if task in ("pick_place", "insert") else data.mocap_pos[0].copy()
    if return_action:
        return p, actions
    return p

def classify(pxy, target, wrongs, pz=None, insert=False):
    all_xy = np.vstack([target, wrongs])
    d = np.linalg.norm(all_xy - pxy[None], axis=1)
    nearest = int(np.argmin(d))
    radius = 0.032 if insert else R
    if insert and pz is not None and float(pz) > 0.055:
        return ("drop", 0)
    if d[nearest] < radius:
        return ("success", 2) if nearest == 0 else ("wrong_target", 1)
    return ("drop", 0)

def geom_for_mode(endpoint, rng, mode):
    # Continuous target coordinates.  The mode is not exposed to the model.
    centers = DIRS * .16
    d = int(np.argmin(np.linalg.norm(centers - endpoint[None], axis=1)))
    if mode == 0:  # success: target near this action endpoint
        target = endpoint + rng.uniform(-.018, .018, 2)
        wrongs = centers[[(d + 1) % 3, (d + 2) % 3]] + rng.uniform(-.018, .018, (2, 2))
    elif mode == 1:  # wrong target: a distractor is near endpoint
        wrongs = np.array([endpoint + rng.uniform(-.012, .012, 2),
                           centers[(d + 2) % 3] + rng.uniform(-.012, .012, 2)])
        target = centers[(d + 1) % 3] + rng.uniform(-.012, .012, 2)
    else:  # drop: every candidate is outside the acceptance disk
        offsets = rng.uniform(-.06, .06, (3, 2))
        target = centers[0] + offsets[0]
        wrongs = np.array([centers[1] + offsets[1], centers[2] + offsets[2]])
        # Reject accidental near points.
        while np.min(np.linalg.norm(np.vstack([target, wrongs]) - endpoint, axis=1)) < .06:
            offsets = rng.uniform(-.08, .08, (3, 2))
            target = centers[0] + offsets[0]
            wrongs = np.array([centers[1] + offsets[1], centers[2] + offsets[2]])
    return target, wrongs

def fact_policy_rows(n=240, seed=2027, mixed_amplitude=False, task="reach"):
    """FACT recipe under MuJoCo: expert successes + wrong-direction policy misses."""
    rng = np.random.default_rng(seed)
    mujoco, model, data = make_world()
    rows = []
    half = n // 2
    for i in range(n):
        payload = rng.uniform(-.02, .02, 2)
        goal = int(rng.integers(3))
        target = payload + DIRS[goal] * .16 + rng.uniform(-.008, .008, 2)
        wrongs = np.vstack([
            payload + DIRS[(goal + 1) % 3] * .16,
            payload + DIRS[(goal + 2) % 3] * .16,
        ])
        if i < half:
            direction = goal
            amp = .16
        else:
            direction = int((goal + 1 + int(rng.integers(2))) % 3)
            amp = float(rng.uniform(.24, .40)) if mixed_amplitude else .16
        p, action = rollout(mujoco, model, data, task, direction, amp, payload, return_action=True)
        label, yi = classify(p[:2], target, wrongs)
        rows.append({
            "task": task, "direction": int(direction), "amplitude": float(amp),
            "label": label, "y": int(yi), "success": int(label == "success"),
            "action": action, "target_xy": target.tolist(),
            "wrong_xy": wrongs.reshape(-1).tolist(), "payload_xy": payload.tolist(),
            "endpoint_xy": p[:2].tolist(), "recipe": (
                "fact_policy_mixed" if mixed_amplitude else "fact_policy_matched"),
        })
    return rows


def paired_rows(n, seed=2027):
    rng = np.random.default_rng(seed); mujoco, model, data = make_world(); rows=[]
    # Factorial layout: reuse each exact geometry for all three actions.  This
    # makes geometry-only prediction impossible in principle: the same layout
    # appears with success, wrong-target and drop outcomes under different
    # directions.  The task identity is also balanced independently.
    tasks = ("reach", "push", "pick_place", "insert")
    groups = max(1, n // (len(tasks) * 3))
    for task in tasks:
        for g in range(groups):
            payload = rng.uniform(-.02, .02, 2)
            goal = int(rng.integers(3)); wrong_dir = (goal + int(rng.integers(1, 3))) % 3
            drop_dir = 3 - goal - wrong_dir
            target = payload + DIRS[goal] * .16 + rng.uniform(-.012, .012, 2)
            wrong1 = payload + DIRS[wrong_dir] * .16 + rng.uniform(-.012, .012, 2)
            wrong2 = payload + DIRS[drop_dir] * .16 + np.array([.075, .075])
            wrongs = np.vstack([wrong1, wrong2])
            for d in range(3):
                p = rollout(mujoco, model, data, task, d, .16, payload)
                label, yi = classify(p[:2], target, wrongs, pz=p[2], insert=(task=="insert"))
                rows.append({"task":task,"direction":d,"amplitude":.16,"label":label,"y":yi,
                             "target_xy":target.tolist(),"wrong_xy":wrongs.reshape(-1).tolist(),
                             "payload_xy":payload.tolist(),"endpoint_xy":p[:2].tolist(),
                             "endpoint_z":float(p[2]),"layout_id":int(g)})
    return rows[:n]

def confounded_rows(n, seed=11):
    rng=np.random.default_rng(seed); mujoco, model, data=make_world(); rows=[]
    target=np.array([.16,0.])
    for i in range(n):
        amp=.08 if i%2 else .16; p=rollout(mujoco,model,data,"reach",0,amp,np.zeros(2))
        label,yi=classify(p[:2],target,np.array([[0,.16],[0,-.16]]))
        rows.append({"task":"reach","direction":0,"amplitude":amp,"label":label,"y":yi,
                     "target_xy":target.tolist(),"wrong_xy":[0,.16,0,-.16],"payload_xy":[0,0],"endpoint_xy":p[:2].tolist()})
    return rows

def feat(r, action=True):
    a=np.eye(3)[r["direction"]] if action else np.empty(0)
    return np.r_[a, r["amplitude"], r["target_xy"], r["wrong_xy"], r["payload_xy"],
                 [int(r["task"]=="push"), int(r["task"]=="pick_place"), int(r["task"]=="insert")]]

def evaluate(train, test, name):
    X=np.stack([feat(r) for r in train]); y=np.array([r["y"] for r in train])
    Xt=np.stack([feat(r) for r in test]); yt=np.array([r["y"] for r in test])
    clf=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1200,random_state=0).fit(X,y)
    proba=clf.predict_proba(Xt); pred=clf.classes_[np.argmax(proba,axis=1)]
    success_col=int(np.flatnonzero(clf.classes_==2)[0]) if 2 in clf.classes_ else -1
    success_prob=proba[:,success_col] if success_col >= 0 else np.zeros(len(yt))
    binary_auc=float(roc_auc_score((yt==2).astype(int),success_prob))
    if len(clf.classes_) == 3:
        auc=float(roc_auc_score(yt,proba,multi_class="ovr",average="macro"))
        out={"task":"three_class","accuracy":float(accuracy_score(yt,pred)),"macro_f1":float(f1_score(yt,pred,average="macro")),"macro_auroc":auc,"binary_success_auroc":binary_auc}
    else:
        # The perturbation set intentionally contains only success/drop.  Its
        # fair comparison is binary success-vs-non-success on the same test.
        success_col=int(np.flatnonzero(clf.classes_==2)[0]) if 2 in clf.classes_ else -1
        p=proba[:,success_col] if success_col >= 0 else np.zeros(len(yt))
        ybin=(yt==2).astype(int)
        out={"task":"binary_success","accuracy":float(accuracy_score(ybin,(p>=.5).astype(int))),"auroc":float(roc_auc_score(ybin,p)),"binary_success_auroc":binary_auc}
    return out

def layout_groups(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], int(r.get("layout_id", -1))), []).append(r)
    return [cands for cands in groups.values() if len(cands) == 3]


def _fit_success_scorer(train, kind):
    if kind == "paired":
        X = np.stack([feat(r) for r in train])
        y = np.array([r["y"] for r in train])
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1200, random_state=0).fit(X, y)
        col = int(np.flatnonzero(clf.classes_ == 2)[0]) if 2 in clf.classes_ else None
        def score(rows):
            if col is None:
                return np.zeros(len(rows))
            return clf.predict_proba(np.stack([feat(r) for r in rows]))[:, col]
        return score
    if kind == "perturbation":
        X = np.stack([feat(r) for r in train])
        y = np.array([int(r["label"] == "success" or r.get("y") == 2) for r in train])
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        pos = 1 if 1 in clf.classes_ else int(clf.classes_[-1])
        col = int(np.flatnonzero(clf.classes_ == pos)[0])
        def score(rows):
            return clf.predict_proba(np.stack([feat(r) for r in rows]))[:, col]
        return score
    if kind == "action_only":
        X = np.stack([feat(r)[:4] for r in train])
        y = np.array([int(r["label"] == "success") for r in train])
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        col = list(clf.classes_).index(1) if 1 in clf.classes_ else None
        def score(rows):
            if col is None:
                return np.zeros(len(rows))
            return clf.predict_proba(np.stack([feat(r)[:4] for r in rows]))[:, col]
        return score
    raise ValueError(kind)


def closed_loop(train, test, confound, n_boot=400, seed=0):
    layouts = layout_groups(test)
    oracle = np.array([any(r["label"] == "success" for r in c) for c in layouts], np.float64)
    random_rate = float(np.mean([
        np.mean([r["label"] == "success" for r in c]) for c in layouts
    ])) if layouts else float("nan")
    scorers = {
        "paired": _fit_success_scorer(train, "paired"),
        "perturbation": _fit_success_scorer(confound, "perturbation"),
        "action_only": _fit_success_scorer(train, "action_only"),
    }

    def pick(name):
        hits = []
        score_fn = scorers[name]
        for cands in layouts:
            scores = score_fn(cands)
            chosen = cands[int(np.argmax(scores))]
            hits.append(int(chosen["label"] == "success"))
        return np.asarray(hits, np.float64)

    rates = {
        "oracle": float(oracle.mean()) if len(oracle) else float("nan"),
        "random": random_rate,
        "paired": float(pick("paired").mean()) if layouts else float("nan"),
        "perturbation": float(pick("perturbation").mean()) if layouts else float("nan"),
        "action_only": float(pick("action_only").mean()) if layouts else float("nan"),
    }
    paired_hits = pick("paired") if layouts else np.array([])
    perturb_hits = pick("perturbation") if layouts else np.array([])
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(paired_hits)
    for _ in range(n_boot):
        if n == 0:
            break
        b = rng.integers(0, n, n)
        deltas.append(float(paired_hits[b].mean() - perturb_hits[b].mean()))
    if deltas:
        q = np.percentile(deltas, [2.5, 50, 97.5])
        rates["paired_minus_perturbation"] = {
            "median": float(q[1]), "ci95": [float(q[0]), float(q[2])],
            "n_layouts": n,
        }
    else:
        rates["paired_minus_perturbation"] = {"median": float("nan"), "ci95": [float("nan"), float("nan")], "n_layouts": 0}
    per_task = {}
    for task in sorted({c[0]["task"] for c in layouts}):
        idx = [i for i, c in enumerate(layouts) if c[0]["task"] == task]
        if not idx:
            continue
        per_task[task] = {
            "n_layouts": len(idx),
            "paired": float(paired_hits[idx].mean()),
            "perturbation": float(perturb_hits[idx].mean()),
            "oracle": float(oracle[idx].mean()),
        }
    rates["per_task"] = per_task
    return rates


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=900)
    ap.add_argument("--output",required=True)
    ap.add_argument("--mode", choices=["benchmark", "fact-policy"], default="benchmark")
    args=ap.parse_args()
    if args.mode == "fact-policy":
        matched = fact_policy_rows(args.n, mixed_amplitude=False)
        mixed = fact_policy_rows(args.n, mixed_amplitude=True)
        out = {
            "matched": matched,
            "mixed": mixed,
            "matched_success_rate": float(np.mean([r["success"] for r in matched])),
            "mixed_success_rate": float(np.mean([r["success"] for r in mixed])),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out), encoding="utf-8")
        print(json.dumps({k: out[k] for k in ("matched_success_rate", "mixed_success_rate",)}, indent=2))
        return
    rows=paired_rows(args.n); conf=confounded_rows(args.n)
    # Split by layout episode, never by individual action row.  Otherwise two
    # actions from one geometry could leak into train/test.
    rng=np.random.default_rng(2028)
    groups=np.array([f"{r['task']}_{r.get('layout_id', i)}" for i,r in enumerate(rows)])
    ug=rng.permutation(np.unique(groups)); gcut=len(ug)//2
    tr_groups=set(ug[:gcut]); tr_idx=[i for i,g in enumerate(groups) if g in tr_groups]; te_idx=[i for i,g in enumerate(groups) if g not in tr_groups]
    train,test=[rows[i] for i in tr_idx],[rows[i] for i in te_idx]
    paired=evaluate(train,test,"paired")
    perturb=evaluate(conf,test,"perturbation")
    # Action-only and geometry-only diagnostics on the paired test split.
    Xa=np.stack([feat(r,action=True)[:4] for r in train]); Xat=np.stack([feat(r,action=True)[:4] for r in test])
    # Geometry-only is intentionally restricted to the target pose and task
    # context.  Including the two distractor slots would expose the sampled
    # class-construction pattern (a dataset artifact), rather than testing
    # whether geometry helps resolve the action direction.
    def gfeat(r):
        return np.r_[r["target_xy"], r["payload_xy"]]
    Xg=np.stack([gfeat(r) for r in train]); Xgt=np.stack([gfeat(r) for r in test]); y=np.array([r["y"] for r in train]); yt=np.array([r["y"] for r in test])
    def au(X,Xt):
        c=LogisticRegression(max_iter=2000).fit(X,y); p=c.predict_proba(Xt); return float(roc_auc_score(yt,p,multi_class="ovr",average="macro"))
    out={"n":len(rows),"tasks":{t:sum(r["task"]==t for r in rows) for t in ("reach","push","pick_place","insert")},
         "class_counts":{k:sum(r["label"]==k for r in rows) for k in ("success","wrong_target","drop")},
         "paired_test":{"action_only_macro_auroc":au(Xa,Xat),"geometry_only_macro_auroc":au(Xg,Xgt),"paired_predictor":paired,"perturbation_predictor":perturb}}
    out["paired_test"]["paired_minus_perturbation_auroc"]=paired["binary_success_auroc"]-perturb["binary_success_auroc"]
    out["closed_loop"]=closed_loop(train,test,conf)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
