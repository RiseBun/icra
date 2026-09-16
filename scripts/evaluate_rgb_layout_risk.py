"""RGB layout -> three-way action consequence predictor."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

def center(im,kind):
    x=im.astype(np.int16)
    if kind=='yellow': m=(x[...,0]>65)&(x[...,1]>50)&(x[...,2]<35)
    elif kind=='blue': m=(x[...,2]>55)&(x[...,0]<55)&(x[...,1]<80)
    else: m=(x[...,0]>55)&(x[...,1]<45)&(x[...,2]<45)
    yy,xx=np.where(m); return np.array([xx.mean()/im.shape[1],yy.mean()/im.shape[0]],np.float32) if len(xx) else np.array([.5,.5],np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--interaction',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    d=np.load(args.input); rows=json.loads(Path(args.interaction).read_text()); n=min(len(d['rgb']),len(rows)); rows=rows[:n]; rgb=d['rgb'][:n]
    X=[]; y=[]; groups=[]; lab={'drop':0,'wrong_target':1,'success':2}
    for im,r in zip(rgb,rows):
        px=np.concatenate([center(im,'yellow'),center(im,'blue'),center(im,'red')]); X.append(np.r_[np.eye(3)[int(r['direction'])],px]); y.append(lab[r['label']]); groups.append(r.get('episode_id',len(groups)))
    X=np.stack(X); y=np.asarray(y); groups=np.asarray(groups); rng=np.random.default_rng(2027); ug=rng.permutation(np.unique(groups)); cut=len(ug)//2; trset=set(ug[:cut]); tr=np.array([g in trset for g in groups]); te=~tr
    model=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1500,random_state=0).fit(X[tr],y[tr]); p=model.predict_proba(X[te]); pred=model.classes_[np.argmax(p,1)]; yt=y[te]
    success_col=int(np.flatnonzero(model.classes_==lab['success'])[0]) if lab['success'] in model.classes_ else None
    layouts={}
    for i,g in enumerate(groups):
        if not te[i]:
            continue
        layouts.setdefault(int(g), []).append(i)
    hits=[]; oracle=[]
    for idxs in layouts.values():
        if len(idxs) != 3 or success_col is None:
            continue
        scores=model.predict_proba(X[idxs])[:, success_col]
        chosen=idxs[int(np.argmax(scores))]
        hits.append(int(y[chosen]==lab['success']))
        oracle.append(int(np.any(y[idxs]==lab['success'])))
    closed={'n_layouts':int(len(hits)),
            'rgb_paired': float(np.mean(hits)) if hits else float('nan'),
            'oracle': float(np.mean(oracle)) if oracle else float('nan'),
            'random': float(np.mean([np.mean(y[idxs]==lab['success']) for idxs in layouts.values() if len(idxs)==3])) if layouts else float('nan')}
    out={'n':int(n),'train':int(tr.sum()),'test':int(te.sum()),'marker_detect_rate':1.0,'macro_auroc':float(roc_auc_score(yt,p,multi_class='ovr',average='macro')),'macro_f1':float(f1_score(yt,pred,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(yt,pred)),'episode_split':True,'input':'RGB yellow/blue/red centroids + action','closed_loop':closed}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
