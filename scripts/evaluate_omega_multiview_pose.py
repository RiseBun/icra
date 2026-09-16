"""Evaluate Omega multi-view tokens with camera-pose conditioning."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

POSE=np.array([[0.,-.68,.62],[-.62,-.35,.55],[.62,-.35,.55]],np.float32)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--omega',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    d=np.load(args.input); o=np.load(args.omega); n=min(len(d['seat_positions']),len(o['scene_tokens'])); y=d['seat_positions'][:n].astype(np.float32); z=o['scene_tokens'][:n].astype(np.float32); pose=np.broadcast_to(POSE[None,:z.shape[1],:],(n,z.shape[1],3)).reshape(n,-1); z=z.reshape(n,-1); idx=np.random.default_rng(2027).permutation(n); cut=n//2; tr,te=idx[:cut],idx[cut:]
    out={"n":int(n),"views":int(o['scene_tokens'].shape[1]),"split":"random_half","omega_frozen":True}
    for name,X in [("omega_only",z),("omega_pose",np.c_[z,pose])]:
        k=min(32,len(tr)-1); m=make_pipeline(StandardScaler(),PCA(k,random_state=0),Ridge(alpha=1000)).fit(X[tr],y[tr]); p=m.predict(X[te]); out[name+'_rmse_m']=float(np.sqrt(mean_squared_error(y[te],p)))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
