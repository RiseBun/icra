"""Closed-loop active observation driven by the RGB risk predictor."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier

DIRS=np.array([[1.,0.],[-.5,.8660254],[-.5,-.8660254]],np.float32); lab={'drop':0,'wrong_target':1,'success':2}
def center(im,kind):
    x=im.astype(np.int16); m=(x[...,0]>65)&(x[...,1]>50)&(x[...,2]<35) if kind=='yellow' else (x[...,2]>55)&(x[...,0]<55)&(x[...,1]<80) if kind=='blue' else (x[...,0]>55)&(x[...,1]<45)&(x[...,2]<45); yy,xx=np.where(m); return np.array([xx.mean()/im.shape[1],yy.mean()/im.shape[0]],np.float32) if len(xx) else np.array([.5,.5],np.float32)
def pixfeat(im,d,noise,rng):
    p=np.concatenate([center(im,'yellow'),center(im,'blue'),center(im,'red')]); p=np.clip(p+rng.normal(0,noise,6),0,1); return np.r_[np.eye(3)[d],p]
def truth(r,d):
    seats=np.asarray(r['seat_positions'],np.float32).reshape(3,2); target=np.asarray(r['target_xy'],np.float32)
    # Evaluation derives the target index geometrically from the seat layout;
    # the predictor never receives this integer label.  The v6 factor task
    # defines candidate d as the action toward seat d.
    target_idx=int(np.argmin(np.linalg.norm(seats-target[None],axis=1)))
    return 2 if int(d)==target_idx else 0
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--interaction',required=True); ap.add_argument('--output',required=True); ap.add_argument('--first-noise',type=float,default=.08); ap.add_argument('--second-noise',type=float,default=.005); args=ap.parse_args(); d=np.load(args.input); rows=json.loads(Path(args.interaction).read_text()); n=min(len(rows),len(d['rgb'])); rows=rows[:n]; rgb=d['rgb'][:n]; rng=np.random.default_rng(2027)
    X=[]; y=[]
    for im,r in zip(rgb,rows): X.append(pixfeat(im,int(r['direction']),0.,rng)); y.append(lab[r['label']])
    X=np.stack(X); y=np.asarray(y); idx=rng.permutation(n); cut=n//2; tr,te=idx[:cut],idx[cut:]; clf=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1500,random_state=0).fit(X[tr],y[tr]); stats={'fixed_view':[],'risk_active':[]}; obs={'fixed_view':[],'risk_active':[]}
    for i in te:
        r=rows[i]; im=rgb[i];
        def choose(noise):
            feats=np.stack([pixfeat(im,d,noise,rng) for d in range(3)]); p=clf.predict_proba(feats); risk=p[:,0]+p[:,1]; a=int(np.argmin(risk)); return a,float(risk[a])
        a0,r0=choose(args.first_noise); a1,_=choose(args.second_noise); use=r0>.45; af=a1 if use else a0; stats['fixed_view'].append(int(truth(r,a0)==2)); stats['risk_active'].append(int(truth(r,af)==2)); obs['fixed_view'].append(0); obs['risk_active'].append(int(use))
    out={'n_test':len(te),'success_rate':{k:float(np.mean(v)) for k,v in stats.items()},'mean_extra_views':{k:float(np.mean(v)) for k,v in obs.items()},'utility_lambda_0.10':{k:float(np.mean(stats[k])-.1*np.mean(obs[k])) for k in stats},'first_view_noise_px':args.first_noise,'second_view_noise_px':args.second_noise,'threshold':.45,'input':'RGB layout risk predictor'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
