"""Recover all three candidate positions from explicit RGB colors."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

def center(im, kind):
    x=im.astype(np.int16)
    if kind=='yellow': m=(x[...,0]>65)&(x[...,1]>50)&(x[...,2]<35)
    elif kind=='blue': m=(x[...,2]>55)&(x[...,0]<55)&(x[...,1]<80)
    else: m=(x[...,0]>55)&(x[...,1]<45)&(x[...,2]<45)
    yy,xx=np.where(m)
    return (np.array([xx.mean()/im.shape[1],yy.mean()/im.shape[0]],np.float32),len(xx)>3) if len(xx) else (np.array([.5,.5],np.float32),False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--interaction',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    d=np.load(args.input); rows=json.loads(Path(args.interaction).read_text()); rgb=d['rgb']; n=min(len(rgb),len(rows)); rgb=rgb[:n]; rows=rows[:n]
    pix=[]; world=[]; valid=[]
    for im,r in zip(rgb,rows):
        seats=np.asarray(r['seat_positions'],np.float32).reshape(3,2); target=int(r['target']); others=[i for i in range(3) if i!=target]
        for kind,idx in [('yellow',target),('blue',others[0]),('red',others[1])]:
            c,ok=center(im,kind); pix.append(c); world.append(seats[idx]); valid.append(ok)
    pix=np.stack(pix); world=np.stack(world); valid=np.asarray(valid); idx=np.arange(n*3); rng=np.random.default_rng(2027); rng.shuffle(idx); cut=len(idx)//2; tr,te=idx[:cut],idx[cut:]
    reg=make_pipeline(PolynomialFeatures(2),Ridge(alpha=1e-3)).fit(pix[tr],world[tr]); pred=reg.predict(pix[te]); rmse=float(np.sqrt(np.mean((pred-world[te])**2)))
    out={'n_images':int(n),'valid_color_rate':float(valid.mean()),'all_layout_rmse_m':rmse,'calibration':'quadratic_ridge','split':'random_half'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
