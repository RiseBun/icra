"""Calibration and risk-coverage metrics for the multiclass core predictor."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

def ece(y,p,bins=10):
    conf=p.max(1); pred=p.argmax(1); out=0.0
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        m=(conf>=lo)&(conf<hi if hi<1 else conf<=hi)
        if m.any(): out += m.mean()*abs(conf[m].mean()-(pred[m]==y[m]).mean())
    return float(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--interaction",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
    rows=json.loads(Path(args.interaction).read_text()); lab={"drop":0,"wrong_target":1,"success":2}; y=np.array([lab[r["label"]] for r in rows]); a=np.c_[np.eye(3)[[r["direction"] for r in rows]],[r["amplitude"] for r in rows]]; g=np.array([r["target_xy"]+r["seat_positions"]+r["payload_xy"] for r in rows]); X=np.c_[a,g]; idx=np.random.default_rng(2027).permutation(len(y)); cut=len(y)//2; tr,te=idx[:cut],idx[cut:]
    raw=HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=31,random_state=0).fit(X[tr],y[tr])
    p_raw=raw.predict_proba(X[te])
    clf=CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=31,random_state=0), method="sigmoid", cv=3).fit(X[tr], y[tr])
    p=clf.predict_proba(X[te]); pred=p.argmax(1); conf=p.max(1); correct=(pred==y[te]).astype(float)
    coverage=[]
    for cov in [.1,.2,.4,.6,.8,1.0]:
        k=max(1,int(len(te)*cov)); keep=np.argsort(-conf)[:k]; coverage.append({"coverage":cov,"accuracy":float(correct[keep].mean()),"risk":float(1-correct[keep].mean()),"mean_confidence":float(conf[keep].mean())})
    onehot=np.eye(3)[y[te]]; brier=float(np.mean(np.sum((p-onehot)**2,axis=1)))
    result={"n_test":len(te),"accuracy":float(accuracy_score(y[te],pred)),"brier_multiclass":brier,"ece":ece(y[te],p),"raw_ece":ece(y[te],p_raw),"risk_coverage":coverage}
    Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
