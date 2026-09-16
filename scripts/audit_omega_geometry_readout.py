"""Evaluate whether frozen Omega tokens retain randomized seat geometry."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True)
    ap.add_argument("--omega",required=True); ap.add_argument("--output",required=True)
    args=ap.parse_args(); d=np.load(args.input); o=np.load(args.omega)
    y=d["seat_positions"].astype(np.float32)
    n=min(len(y),len(o["scene_tokens"]))
    y=y[:n]
    z=o["scene_tokens"].astype(np.float32)[:n].reshape(n,-1)
    idx=np.random.default_rng(0).permutation(len(y)); cut=max(1,len(y)//2); tr,te=idx[:cut],idx[cut:]
    model=Ridge(alpha=10.0).fit(z[tr],y[tr]); pred=model.predict(z[te])
    rmse=float(np.sqrt(mean_squared_error(y[te],pred)))
    # Relative geometry is centered to remove global camera/scene translation.
    yc=y.reshape(len(y),3,2); pc=pred.reshape(len(pred),3,2)
    rel_rmse=float(np.sqrt(np.mean(((yc[te]-yc[te].mean(1,keepdims=True))-(pc-pc.mean(1,keepdims=True)))**2)))
    result={"n":int(len(y)),"token_dim":int(z.shape[1]),"seat_rmse_m":rmse,
            "relative_seat_rmse_m":rel_rmse,"split":"random_half","frozen":True}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
