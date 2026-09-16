"""Three-way success/wrong-target/drop core evaluation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.ensemble import HistGradientBoostingClassifier

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--interaction",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
    rows=json.loads(Path(args.interaction).read_text()); labels=["drop","wrong_target","success"]
    y=np.asarray([labels.index(r["label"]) for r in rows]); action=np.c_[np.eye(3)[[r["direction"] for r in rows]], [r["amplitude"] for r in rows]]
    geom=np.asarray([r["target_xy"]+r["seat_positions"]+r["payload_xy"] for r in rows])
    idx=np.random.default_rng(2027).permutation(len(y)); cut=len(y)//2; tr,te=idx[:cut],idx[cut:]
    table={}
    for name,X,nl in [("action_only",action,False),("geometry_only",geom,False),("action_plus_geometry",np.c_[action,geom],True)]:
        base=HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, random_state=0) if nl else LogisticRegression(max_iter=3000)
        clf=make_pipeline(StandardScaler(),base).fit(X[tr],y[tr]); p=clf.predict_proba(X[te]); pred=clf.predict(X[te])
        table[name]={"accuracy":float(accuracy_score(y[te],pred)),"balanced_accuracy":float(balanced_accuracy_score(y[te],pred)),"macro_f1":float(f1_score(y[te],pred,average="macro")),"macro_auroc":float(roc_auc_score(y[te],p,multi_class="ovr",average="macro")),"confusion":confusion_matrix(y[te],pred).tolist()}
    result={"n":len(y),"class_counts":{k:int((y==i).sum()) for i,k in enumerate(labels)},"labels":labels,"split":{"train":len(tr),"test":len(te)},"table":table}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
