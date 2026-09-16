"""Occlusion-aware RGB risk gate for active observation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier

DIRS=np.array([[1.,0.],[-.5,.8660254],[-.5,-.8660254]],np.float32); lab={'drop':0,'wrong_target':1,'success':2}
def center(im,kind):
    x=im.astype(np.int16); m=(x[...,0]>65)&(x[...,1]>50)&(x[...,2]<35) if kind=='yellow' else (x[...,2]>55)&(x[...,0]<55)&(x[...,1]<80) if kind=='blue' else (x[...,0]>55)&(x[...,1]<45)&(x[...,2]<45); yy,xx=np.where(m); return (np.array([xx.mean()/im.shape[1],yy.mean()/im.shape[0]],np.float32),len(xx)>3) if len(xx) else (np.array([.5,.5],np.float32),False)
def feat(im,d): return np.r_[np.eye(3)[d],center(im,'yellow')[0],center(im,'blue')[0],center(im,'red')[0]]
def truth(r,d):
    seats=np.asarray(r['seat_positions'],np.float32).reshape(3,2); target=np.asarray(r['target_xy'],np.float32); ti=int(np.argmin(np.linalg.norm(seats-target[None],axis=1))); return 2 if int(d)==ti else 0
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--interaction',required=True); ap.add_argument('--output',required=True); args=ap.parse_args(); d=np.load(args.input); rows=json.loads(Path(args.interaction).read_text()); n=min(len(rows),len(d['rgb'])); rgb=d['rgb'][:n]; rows=rows[:n]; rng=np.random.default_rng(2027); X=np.stack([feat(im,r['direction']) for im,r in zip(rgb,rows)]); y=np.asarray([lab[r['label']] for r in rows]); idx=rng.permutation(n); cut=n//2; tr,te=idx[:cut],idx[cut:]; clf=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1500,random_state=0).fit(X[tr],y[tr]); result={'n_test':len(te),'levels':{}}
    for frac in [0.,.10,.20,.30,.40]:
        fixed=[]; active=[]; obs=[]
        for i in te:
            im=rgb[i].copy(); h,w=im.shape[:2]; side=int(np.sqrt(frac)*min(h,w))
            if side>0:
                x0=int(rng.integers(0,max(1,w-side+1))); y0=int(rng.integers(0,max(1,h-side+1))); im[y0:y0+side,x0:x0+side]=24
            valid=np.mean([center(im,k)[1] for k in ('yellow','blue','red')]); batch=np.stack([feat(im,j) for j in range(3)]); p=clf.predict_proba(batch); risk=p[:,0]+p[:,1]; a0=int(np.argmin(risk));
            clean=np.stack([feat(rgb[i],j) for j in range(3)]); pc=clf.predict_proba(clean); ac=int(np.argmin(pc[:,0]+pc[:,1])); use=bool(risk[a0]>.45 or valid<.67); af=ac if use else a0; fixed.append(int(truth(rows[i],a0)==2)); active.append(int(truth(rows[i],af)==2)); obs.append(int(use))
        result['levels'][str(frac)]={'fixed_success':float(np.mean(fixed)),'active_success':float(np.mean(active)),'mean_extra_views':float(np.mean(obs)),'fixed_utility':float(np.mean(fixed)),'active_utility':float(np.mean(active)-.1*np.mean(obs))}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
