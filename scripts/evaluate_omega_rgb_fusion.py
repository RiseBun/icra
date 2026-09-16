"""Compare frozen Omega, explicit RGB layout, and a lightweight fusion adapter."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def center(im,kind):
    x=im.astype(np.int16)
    if kind=='yellow': m=(x[...,0]>65)&(x[...,1]>50)&(x[...,2]<35)
    elif kind=='blue': m=(x[...,2]>55)&(x[...,0]<55)&(x[...,1]<80)
    else: m=(x[...,0]>55)&(x[...,1]<45)&(x[...,2]<45)
    yy,xx=np.where(m); return np.array([xx.mean()/im.shape[1],yy.mean()/im.shape[0]],np.float32) if len(xx) else np.array([.5,.5],np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--interaction',required=True); ap.add_argument('--omega',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    d=np.load(args.input); o=np.load(args.omega); rows=json.loads(Path(args.interaction).read_text()); n=min(len(rows),len(o['scene_tokens']),len(d['rgb'])); rows=rows[:n]
    y=np.stack([np.asarray(r['seat_positions'],np.float32) for r in rows]); z=o['scene_tokens'][:n].astype(np.float32).reshape(n,-1)
    rgb=[]; valid=[]
    for im,r in zip(d['rgb'][:n],rows):
        t=int(r['target']); others=[i for i in range(3) if i!=t]; seats=np.asarray(r['seat_positions'],np.float32).reshape(3,2)
        cs=[center(im,'yellow'),center(im,'blue'),center(im,'red')]
        ordered=[None,None,None]; ordered[t]=cs[0]; ordered[others[0]]=cs[1]; ordered[others[1]]=cs[2]
        rgb.append(np.concatenate(ordered)); valid.append(True)
    rgb=np.stack(rgb); idx=np.random.default_rng(2027).permutation(n); cut=n//2; tr,te=idx[:cut],idx[cut:]
    # Ordered RGB target is enough for this controlled camera; color layout
    # serves as a high-quality visual baseline.
    def fit_eval(model,X,Xt):
        model.fit(X[tr],y[tr]); p=model.predict(Xt[te]); return float(np.sqrt(mean_squared_error(y[te],p)))
    omega=fit_eval(make_pipeline(StandardScaler(),PCA(n_components=min(20,len(tr)-1),random_state=0),Ridge(alpha=10)),z,z)
    rgb_rmse=fit_eval(make_pipeline(PolynomialFeatures(2),Ridge(alpha=1e-3)),rgb,rgb)
    # Keep the low-dimensional RGB geometry branch intact; applying PCA to
    # the concatenation would erase it because Omega has ~35k dimensions.
    pca=PCA(n_components=min(20,len(tr)-1),random_state=0).fit(StandardScaler().fit_transform(z[tr]))
    zs=StandardScaler().fit(z[tr]); ztr=pca.transform(zs.transform(z[tr])); zte=pca.transform(zs.transform(z[te]))
    fusion_model=Ridge(alpha=10).fit(np.c_[ztr,rgb[tr]],y[tr]); fusion_pred=fusion_model.predict(np.c_[zte,rgb[te]])
    fusion=float(np.sqrt(mean_squared_error(y[te],fusion_pred)))
    out={'n':int(n),'omega_only_rmse_m':omega,'rgb_color_layout_rmse_m':rgb_rmse,'omega_rgb_fusion_rmse_m':fusion,'split':'random_half','omega_frozen':True}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
