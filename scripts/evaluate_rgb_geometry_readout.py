"""Evaluate continuous target geometry recovered from rendered RGB."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPClassifier

def centroid(im):
    x=im.astype(np.int16)
    mask=(x[...,0]>80)&(x[...,1]>60)&(x[...,2]<40)
    yy,xx=np.where(mask)
    if len(xx)==0: return np.array([.5,.5]),0
    return np.array([xx.mean()/im.shape[1],yy.mean()/im.shape[0]],np.float32),1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    d=np.load(args.input); rgb=d['rgb']; y=d['success'].astype(int); target=d['target_xy'].astype(np.float32); direction=d['direction'].astype(int)
    cc=[centroid(im) for im in rgb]; X=np.stack([q[0] for q in cc]); det=np.array([q[1] for q in cc])
    idx=np.random.default_rng(2027).permutation(len(y)); cut=len(y)//2; tr,te=idx[:cut],idx[cut:]
    # A quadratic calibration absorbs the fixed camera projection without
    # introducing any simulator geometry into the input.
    reg=make_pipeline(PolynomialFeatures(2),Ridge(alpha=1e-3)).fit(X[tr],target[tr]); pred=reg.predict(X[te])
    rmse=float(np.sqrt(mean_squared_error(target[te],pred)))
    # End-to-end readout: action plus estimated target position.  This is an
    # RGB upper bound, not an Omega result.
    xa=np.eye(3)[direction]
    model=MLPClassifier(hidden_layer_sizes=(32,16),max_iter=1500,random_state=0).fit(np.c_[xa[tr],X[tr]],y[tr])
    auc=float(roc_auc_score(y[te],model.predict_proba(np.c_[xa[te],X[te]])[:,1]))
    out={'n':int(len(y)),'marker_detect_rate':float(det.mean()),'target_xy_rmse_m':rmse,'action_plus_rgb_centroid_auroc':auc,'split':'random_half','calibration':'quadratic_ridge'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
