"""Robustness of the privileged core predictor to geometry noise."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--interaction",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
    rows=json.loads(Path(args.interaction).read_text()); names=["drop","wrong_target","success"]; y=np.array([names.index(r["label"]) for r in rows]); a=np.c_[np.eye(3)[[r["direction"] for r in rows]],[r["amplitude"] for r in rows]]; g=np.array([r["target_xy"]+r["seat_positions"]+r["payload_xy"] for r in rows],np.float32)
    idx=np.random.default_rng(2027).permutation(len(y)); tr,te=idx[:len(y)//2],idx[len(y)//2:]; out={}
    for sigma in [0.0,0.005,0.01,0.02,0.04]:
        rng=np.random.default_rng(9000+int(sigma*10000)); gn=g[te]+rng.normal(0,sigma,g[te].shape).astype(np.float32)
        Xtr=np.c_[a[tr],g[tr]]; Xte=np.c_[a[te],gn]
        clf=HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=31,random_state=0).fit(Xtr,y[tr]); p=clf.predict_proba(Xte); pred=clf.predict(Xte)
        out[str(sigma)]={"macro_auroc":float(roc_auc_score(y[te],p,multi_class="ovr",average="macro")),"macro_f1":float(f1_score(y[te],pred,average="macro")),"balanced_accuracy":float(balanced_accuracy_score(y[te],pred))}
    result={"n":len(y),"noise_units":"meters","split":{"train":len(tr),"test":len(te)},"results":out}; Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
