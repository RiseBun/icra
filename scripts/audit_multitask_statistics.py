"""Statistical audit for the strong-form reach/push factor experiment."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from mujoco_strong_multitask import paired_rows, confounded_rows, feat

def run_task(rows, task, seed):
    rr=[r for r in rows if r["task"]==task]
    groups=np.array([r.get("layout_id",i) for i,r in enumerate(rr)])
    ug=np.random.default_rng(seed).permutation(np.unique(groups)); cut=len(ug)//2
    trset=set(ug[:cut]); tr=[r for r,g in zip(rr,groups) if g in trset]; te=[r for r,g in zip(rr,groups) if g not in trset]
    y=np.array([r["y"] for r in tr]); yt=np.array([r["y"] for r in te]); X=np.stack([feat(r) for r in tr]); Xt=np.stack([feat(r) for r in te])
    Xa=X[:,:4]; Xat=Xt[:,:4]
    def gfeat(r): return np.r_[r["target_xy"],r["payload_xy"]]
    Xg=np.stack([gfeat(r) for r in tr]); Xgt=np.stack([gfeat(r) for r in te])
    def fit_auc(a,b,model):
        c=model.fit(a,y); p=c.predict_proba(b)
        if p.shape[1]==2: return float(roc_auc_score((yt==2).astype(int),p[:,list(c.classes_).index(2)] if 2 in c.classes_ else p[:,-1]))
        return float(roc_auc_score(yt,p,multi_class="ovr",average="macro"))
    joint=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1200,random_state=seed).fit(X,y)
    jp=joint.predict_proba(Xt); joint_auc=float(roc_auc_score(yt,jp,multi_class="ovr",average="macro")); joint_bin=float(roc_auc_score((yt==2).astype(int),jp[:,list(joint.classes_).index(2)]))
    action_auc=fit_auc(Xa,Xat,LogisticRegression(max_iter=2000)); geom_auc=fit_auc(Xg,Xgt,LogisticRegression(max_iter=2000))
    # Direction permutation null: remove action identity from training while
    # preserving the same action marginal distribution.
    yp=np.random.default_rng(seed+100).permutation(Xa[:,0:3],axis=0)
    null_auc=fit_auc(np.c_[yp,Xa[:,3:4]],Xat,LogisticRegression(max_iter=2000))
    # Bootstrap paired AUROC gain over action-only, using test predictions.
    ac=LogisticRegression(max_iter=2000).fit(Xa,y); ap=ac.predict_proba(Xat)[:,1]
    rng=np.random.default_rng(seed+200); bs=[]
    for _ in range(2000):
        j=rng.integers(0,len(yt),len(yt)); yy=(yt[j]==2).astype(int)
        if yy.min()==yy.max(): continue
        bs.append(roc_auc_score(yy,jp[j,list(joint.classes_).index(2)])-roc_auc_score(yy,ap[j]))
    # Perturbation-trained binary baseline evaluated on this same test split.
    conf=confounded_rows(600,seed=seed+300); Xc=np.stack([feat(r) for r in conf]); yc=(np.array([r["y"] for r in conf])==2).astype(int); Xtbin=np.stack([feat(r) for r in te]);
    pc=MLPClassifier(hidden_layer_sizes=(32,16),max_iter=1000,random_state=seed).fit(Xc,yc); pp=pc.predict_proba(Xtbin)[:,1]
    perturb_auc=float(roc_auc_score((yt==2).astype(int),pp))
    return {"n_train":len(tr),"n_test":len(te),"positive_rate":float(np.mean(yt==2)),"action_only_auroc":action_auc,"geometry_only_auroc":geom_auc,"joint_macro_auroc":joint_auc,"joint_binary_success_auroc":joint_bin,"action_permutation_null_auroc":null_auc,"perturbation_binary_auroc":perturb_auc,"joint_minus_action_binary":joint_bin-action_auc,"joint_minus_perturbation_binary":joint_bin-perturb_auc,"bootstrap_joint_minus_action_95ci":[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=900); ap.add_argument("--seeds",type=int,default=5); ap.add_argument("--output",required=True); args=ap.parse_args()
    per={"reach":[],"push":[],"pick_place":[]}
    for s in range(args.seeds):
        rows=paired_rows(args.n,seed=2027+s)
        for t in per: per[t].append(run_task(rows,t,100+s))
    out={"n":args.n,"seeds":args.seeds,"per_task":per}
    for t,v in per.items():
        keys=[k for k in v[0] if isinstance(v[0][k],(int,float))]
        out.setdefault("mean",{})[t]={k:float(np.mean([x[k] for x in v])) for k in keys}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
