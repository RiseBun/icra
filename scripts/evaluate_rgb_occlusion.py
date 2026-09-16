"""Evaluate RGB risk prediction under random rectangular occlusions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,balanced_accuracy_score

def center(im,kind):
    x=im.astype(np.int16); m=(x[...,0]>65)&(x[...,1]>50)&(x[...,2]<35) if kind=='yellow' else (x[...,2]>55)&(x[...,0]<55)&(x[...,1]<80) if kind=='blue' else (x[...,0]>55)&(x[...,1]<45)&(x[...,2]<45); yy,xx=np.where(m); return (np.array([xx.mean()/im.shape[1],yy.mean()/im.shape[0]],np.float32),len(xx)>3) if len(xx) else (np.array([.5,.5],np.float32),False)
def feat(im,d): return np.r_[np.eye(3)[d],center(im,'yellow')[0],center(im,'blue')[0],center(im,'red')[0]]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--interaction',required=True); ap.add_argument('--output',required=True); args=ap.parse_args(); d=np.load(args.input); rows=json.loads(Path(args.interaction).read_text()); n=min(len(rows),len(d['rgb'])); rgb=d['rgb'][:n]; lab={'drop':0,'wrong_target':1,'success':2}; y=np.array([lab[r['label']] for r in rows[:n]]); X=np.stack([feat(im,r['direction']) for im,r in zip(rgb,rows[:n])]); rng=np.random.default_rng(2027); idx=rng.permutation(n); cut=n//2; tr,te=idx[:cut],idx[cut:]; clf=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1500,random_state=0).fit(X[tr],y[tr]); out={'n':n,'test':len(te),'levels':{}}
    for frac in [0.,.10,.20,.30,.40]:
        masked=[]; valid=[]
        for i in te:
            im=rgb[i].copy(); h,w=im.shape[:2]; side=int(np.sqrt(frac)*min(h,w));
            if side>0:
                x0=int(rng.integers(0,max(1,w-side+1))); y0=int(rng.integers(0,max(1,h-side+1))); im[y0:y0+side,x0:x0+side]=24
            cs=[center(im,k)[1] for k in ('yellow','blue','red')]; valid.append(np.mean(cs)); masked.append(feat(im,rows[i]['direction']))
        p=clf.predict_proba(np.stack(masked)); pred=clf.classes_[np.argmax(p,1)]; out['levels'][str(frac)]={'marker_valid_rate':float(np.mean(valid)),'macro_auroc':float(roc_auc_score(y[te],p,multi_class='ovr',average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y[te],pred))}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
