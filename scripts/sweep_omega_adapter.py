"""Sweep low-rank frozen-Omega adapters before considering backbone tuning."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--omega',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    d=np.load(args.input); o=np.load(args.omega); n=min(len(d['seat_positions']),len(o['scene_tokens'])); y=d['seat_positions'][:n].astype(np.float32); z=o['scene_tokens'][:n].astype(np.float32).reshape(n,-1)
    idx=np.random.default_rng(2027).permutation(n); cut=n//2; tr,te=idx[:cut],idx[cut:]; out=[]
    for k in [4,8,12,20,32,40]:
        if k>=len(tr): continue
        for alpha in [0.1,1,10,100,1000]:
            m=make_pipeline(StandardScaler(),PCA(n_components=k,random_state=0),Ridge(alpha=alpha)).fit(z[tr],y[tr]); p=m.predict(z[te]); out.append({'pca':k,'alpha':alpha,'rmse_m':float(np.sqrt(mean_squared_error(y[te],p)))})
    best=min(out,key=lambda x:x['rmse_m']); result={'n':int(n),'best':best,'sweep':out,'split':'random_half','frozen_omega':True}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
