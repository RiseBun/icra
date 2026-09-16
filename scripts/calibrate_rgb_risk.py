"""Calibrate RGB layout risk probabilities on a held-out calibration split."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

def center(im,kind):
    x=im.astype(np.int16)
    m=((x[...,0]>65)&(x[...,1]>50)&(x[...,2]<35)) if kind=='yellow' else ((x[...,2]>55)&(x[...,0]<55)&(x[...,1]<80)) if kind=='blue' else ((x[...,0]>55)&(x[...,1]<45)&(x[...,2]<45))
    yy,xx=np.where(m); return np.array([xx.mean()/im.shape[1],yy.mean()/im.shape[0]],np.float32) if len(xx) else np.array([.5,.5],np.float32)

def ece(y,p,bins=10):
    conf=p.max(1); pred=p.argmax(1); out=0.
    for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(0,1,bins+1)[1:]):
        m=(conf>=lo)&(conf<hi)
        if m.any(): out += m.mean()*abs(np.mean(pred[m]==y[m])-np.mean(conf[m]))
    return float(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--interaction',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    d=np.load(args.input); rows=json.loads(Path(args.interaction).read_text()); n=min(len(rows),len(d['rgb'])); lab={'drop':0,'wrong_target':1,'success':2}; X=[]; y=[]
    for im,r in zip(d['rgb'][:n],rows[:n]): X.append(np.r_[np.eye(3)[int(r['direction'])],center(im,'yellow'),center(im,'blue'),center(im,'red')]); y.append(lab[r['label']])
    X=np.stack(X); y=np.asarray(y); idx=np.random.default_rng(2027).permutation(n); a=int(.6*n); b=int(.8*n); tr,ca,te=idx[:a],idx[a:b],idx[b:]
    clf=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1500,random_state=0).fit(X[tr],y[tr]); pc=clf.predict_proba(X[ca]); pt=clf.predict_proba(X[te]);
    # Multiclass Platt-style calibration: fit a logistic map on log-probability
    # vectors using calibration data only, then apply to the untouched test.
    logits=np.log(np.clip(pc,1e-6,1)); cal=LogisticRegression(multi_class='multinomial',max_iter=2000).fit(logits,y[ca]); pcal=cal.predict_proba(np.log(np.clip(pt,1e-6,1)))
    one=np.eye(3)[y[te]]; brier_raw=float(np.mean(np.sum((pt-one)**2,1))); brier_cal=float(np.mean(np.sum((pcal-one)**2,1)))
    conf=pt.max(1); order=np.argsort(-conf); cov=[]
    for frac in [.2,.4,.6,1.0]:
        k=max(1,int(frac*len(te))); cov.append({'coverage':frac,'accuracy_raw':float(np.mean(pt[order[:k]].argmax(1)==y[te][order[:k]])),'accuracy_cal':float(np.mean(pcal[order[:k]].argmax(1)==y[te][order[:k]]))})
    out={'n':int(n),'split':{'train':int(len(tr)),'calibration':int(len(ca)),'test':int(len(te))},'ece_raw':ece(y[te],pt),'ece_calibrated':ece(y[te],pcal),'brier_raw':brier_raw,'brier_calibrated':brier_cal,'risk_coverage':cov,'input':'RGB color centroids + action','calibration_fit':'calibration split only'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
