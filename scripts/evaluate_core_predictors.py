"""Training-level core comparison on the MuJoCo privileged-geometry set.

This is the main-table experiment independent of Omega perception.  It reports
held-out AUROC for action-only, geometry-only, action+geometry, and the old
amplitude-confounded predictor, with paired bootstrap confidence intervals for
the geometry marginal gain.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def fit_auc(X, y, tr, te, nonlinear=False):
    base = (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                          random_state=0, early_stopping=True)
            if nonlinear else LogisticRegression(max_iter=3000, C=1.0))
    model = make_pipeline(StandardScaler(), base)
    model.fit(X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y[te], p)), float(brier_score_loss(y[te], p))


def bootstrap_delta(Xa, Xg, y, tr, te, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    deltas=[]; auc_a=[]; auc_ag=[]
    # Resample held-out rows; train split remains fixed to avoid leakage.
    for _ in range(n_boot):
        b = rng.integers(0, len(te), len(te)); bt = te[b]
        try:
            a, _ = fit_auc(Xa, y, tr, bt); ag, _ = fit_auc(np.c_[Xa, Xg], y, tr, bt, nonlinear=True)
        except ValueError:
            continue
        deltas.append(ag-a); auc_a.append(a); auc_ag.append(ag)
    q=np.percentile(deltas,[2.5,50,97.5])
    return {"delta_median":float(q[1]),"delta_ci95":[float(q[0]),float(q[2])],
            "bootstrap_samples":len(deltas),"action_auc_median":float(np.median(auc_a)),
            "action_geometry_auc_median":float(np.median(auc_ag))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--interaction",required=True)
    ap.add_argument("--confound",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--bootstrap",type=int,default=1000); args=ap.parse_args()
    inter=json.loads(Path(args.interaction).read_text()); conf=json.loads(Path(args.confound).read_text())
    y=np.asarray([r["success"] for r in inter],np.int64)
    action=np.c_[np.eye(3)[[r["direction"] for r in inter]],
                  np.asarray([r["amplitude"] for r in inter])]
    geom=np.asarray([r["target_xy"]+r["seat_positions"]+r["payload_xy"] for r in inter],np.float64)
    rng=np.random.default_rng(2027); idx=rng.permutation(len(y)); cut=len(y)//2; tr,te=idx[:cut],idx[cut:]
    table={}
    for name,X in (("action_only",action),("geometry_only",geom)):
        auc,brier=fit_auc(X,y,tr,te); table[name]={"auroc":auc,"brier":brier}
    auc,brier=fit_auc(np.c_[action,geom],y,tr,te,nonlinear=True)
    table["action_plus_geometry"]={"auroc":auc,"brier":brier}
    table["action_plus_geometry"]["marginal_gain_over_action"] = table["action_plus_geometry"]["auroc"]-table["action_only"]["auroc"]
    delta=bootstrap_delta(action,geom,y,tr,te,args.bootstrap)
    yc=np.asarray([r["success"] for r in conf],np.int64); xc=np.asarray([[r["amplitude"]] for r in conf])
    cc=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)).fit(xc,yc)
    conf_auc=float(roc_auc_score(yc,cc.predict_proba(xc)[:,1]))
    # Directly test the old perturbation shortcut on the paired test set.  The
    # confounded model only sees amplitude; all paired candidates share one
    # amplitude, so its ranking collapses to chance.
    paired_amp=np.asarray([[r["amplitude"]] for r in inter], np.float64)
    p_pair=cc.predict_proba(paired_amp[te])[:,1]
    try: paired_auc=float(roc_auc_score(y[te],p_pair))
    except ValueError: paired_auc=0.5
    result={"n_interaction":len(y),"n_confound":len(yc),"positive_rate":float(y.mean()),
            "split":{"train":len(tr),"test":len(te),"seed":2027},"table":table,
            "bootstrap_action_geometry_minus_action":delta,
            "amplitude_confound_auc":conf_auc,
            "confound_predictor_on_paired_test_auroc":paired_auc,
            "gate_pass":bool(delta["delta_ci95"][0]>0.10)}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
