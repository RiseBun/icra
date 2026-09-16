"""Evaluate calibration, temperature scaling and selective risk."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,brier_score_loss,confusion_matrix,log_loss,roc_auc_score

def _rows(path):
    d=json.loads(Path(path).read_text()); return d if isinstance(d,list) else d.get('rows',d)
def _scores(rows,key): return np.asarray([r[key] for r in rows],dtype=float)
def _probs(scores,temperature=1.0):
    s=np.asarray(scores,float)
    if s.ndim==1:
        if np.all((s>=0)&(s<=1)):
            q=np.clip(s,1e-6,1-1e-6); z=np.log(q/(1-q))
        else: z=s
        z=z/max(float(temperature),1e-6); return 1/(1+np.exp(-np.clip(z,-40,40)))
    z=s/max(float(temperature),1e-6); z=z-z.max(1,keepdims=True); e=np.exp(np.clip(z,-40,40)); return e/e.sum(1,keepdims=True)
def _nll(labels,probs):
    p=np.asarray(probs); y=np.asarray(labels,int)
    return float(log_loss(y,p,labels=[0,1] if p.ndim==1 else list(range(p.shape[1]))))
def fit_temperature(labels,scores):
    grid=np.exp(np.linspace(np.log(.05),np.log(20.),161)); losses=[_nll(labels,_probs(scores,t)) for t in grid]; return float(grid[int(np.argmin(losses))])
def ece(y,p,bins=10):
    p=np.asarray(p); conf=p.max(1) if p.ndim==2 else np.maximum(p,1-p); pred=p.argmax(1) if p.ndim==2 else (p>=.5).astype(int); out=0.; edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(conf>=lo)&((conf<=hi) if hi==1 else (conf<hi))
        if m.any(): out+=float(m.mean()*abs(conf[m].mean()-(pred[m]==y[m]).mean()))
    return out
def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p); pred=(p>=.5).astype(int) if p.ndim==1 else p.argmax(1)
    if p.ndim==1: out={'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p))}
    else:
        one=np.eye(p.shape[1])[y]; out={'macro_auroc':float(roc_auc_score(y,p,multi_class='ovr',average='macro')),'brier':float(np.mean(np.sum((p-one)**2,1)))}
    out.update({'ece':ece(y,p),'confusion_matrix':confusion_matrix(y,pred).tolist()}); conf=p if p.ndim==1 else p.max(1); cov=[]
    for c in (.1,.2,.4,.6,.8,1.0):
        k=max(1,int(len(y)*c)); keep=np.argsort(-conf)[:k]; cov.append({'coverage':c,'selective_accuracy':float(np.mean(pred[keep]==y[keep])),'risk':float(np.mean(pred[keep]!=y[keep]))})
    out.update({'risk_coverage':cov,'n':int(len(y))}); return out
def conformal_summary(y_cal,p_cal,y_test,p_test,alpha=.1):
    pc=np.asarray(p_cal); pt=np.asarray(p_test); yc=np.asarray(y_cal,int); yt=np.asarray(y_test,int); tc=pc[np.arange(len(yc)),yc] if pc.ndim==2 else np.where(yc==1,pc,1-pc); q=float(np.quantile(1-tc,min(1.,(len(yc)+1)*(1-alpha)/len(yc)),method='higher')); tt=pt[np.arange(len(yt)),yt] if pt.ndim==2 else np.where(yt==1,pt,1-pt); keep=(1-tt<=q); pred=pt.argmax(1) if pt.ndim==2 else (pt>=.5); return {'alpha':alpha,'nonconformity_quantile':q,'coverage':float(keep.mean()),'selective_accuracy':float(np.mean(pred[keep]==yt[keep])) if keep.any() else None,'rejection_rate':float(1-keep.mean())}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input'); ap.add_argument('--train'); ap.add_argument('--calibration'); ap.add_argument('--test'); ap.add_argument('--output',required=True); ap.add_argument('--label-key',default='label'); ap.add_argument('--score-key',default='score'); ap.add_argument('--alpha',type=float,default=.1); a=ap.parse_args()
    if a.input:
        rows=_rows(a.input); y=np.asarray([r[a.label_key] for r in rows],int); s=_scores(rows,a.score_key); out={'mode':'single_input','temperature':1.0,'uncalibrated':metrics(y,_probs(s))}
    else:
        if not (a.train and a.calibration and a.test): raise SystemExit('provide --input or all of --train --calibration --test')
        tr,ca,te=(_rows(x) for x in (a.train,a.calibration,a.test)); del tr; yca=np.asarray([r[a.label_key] for r in ca],int); yte=np.asarray([r[a.label_key] for r in te],int); sca=_scores(ca,a.score_key); ste=_scores(te,a.score_key); temp=fit_temperature(yca,sca); pca=_probs(sca,temp); pte=_probs(ste,temp); out={'mode':'train_calibration_test','temperature':temp,'uncalibrated':metrics(yte,_probs(ste)),'calibrated':metrics(yte,pte),'calibration_nll':_nll(yca,pca),'conformal':conformal_summary(yca,pca,yte,pte,a.alpha)}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
