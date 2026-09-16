"""Dynamic contact version of the action x geometry factor test.

The pusher is velocity controlled; payload endpoints arise from contact,
friction, and mass. Targets are continuous and are never used to construct
the action, so geometry must provide information beyond the action direction.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pybullet as pb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

DIRS=np.array([[1.,0.],[-.5,.8660254],[-.5,-.8660254]])

def run(mass,friction,direction,payload=np.zeros(2),nsteps=80,speed=.30):
    cid=pb.connect(pb.DIRECT); pb.setGravity(0,0,-9.81); pb.setTimeStep(.01)
    plane=pb.createCollisionShape(pb.GEOM_PLANE); pb.createMultiBody(0,plane)
    ec=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.022); pc=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.018)
    ee=pb.createMultiBody(1,ec,basePosition=[0,0,.07]); pay=pb.createMultiBody(mass,pc,basePosition=[*payload,.018])
    pb.changeDynamics(ee,-1,lateralFriction=1.0,rollingFriction=.001); pb.changeDynamics(pay,-1,lateralFriction=friction,rollingFriction=.001)
    d=DIRS[direction]; start=payload-d*.045
    pb.resetBasePositionAndOrientation(ee,[*start,.07],[0,0,0,1]); pb.resetBaseVelocity(pay,[0,0,0],[0,0,0])
    for t in range(nsteps):
        v=speed if 5<=t<55 else 0.
        pb.resetBaseVelocity(ee,[float(d[0]*v),float(d[1]*v),0],[0,0,0]); pb.stepSimulation()
    p,_=pb.getBasePositionAndOrientation(pay); pb.disconnect(cid); return np.asarray(p)

def fit(train,test,mode):
    def f(r):
        a=np.eye(3)[r['direction']]
        if mode=='action': return np.r_[a,r['mass'],r['friction']]
        if mode=='joint': return np.r_[a,r['mass'],r['friction'],r['target'],r['payload']]
        return np.r_[r['target'],r['payload'],r['mass'],r['friction']]
    x=np.stack([f(r) for r in train]); xt=np.stack([f(r) for r in test]); y=np.array([r['class'] for r in train]); yt=np.array([r['class'] for r in test])
    c=MLPClassifier(hidden_layer_sizes=(48,32),max_iter=800,random_state=2027).fit(x,y); p=c.predict_proba(xt); pred=c.classes_[p.argmax(1)]
    vals=[]
    for cidx,cval in enumerate(c.classes_):
        yb=(yt==cval).astype(int)
        if len(np.unique(yb))==2: vals.append(roc_auc_score(yb,p[:,cidx]))
    auc=float(np.mean(vals))
    return {'macro_auroc':float(auc),'accuracy':float(accuracy_score(yt,pred))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n-layouts',type=int,default=120); ap.add_argument('--output',required=True); ap.add_argument('--seed',type=int,default=2027); ap.add_argument('--speed',type=float,default=.30); args=ap.parse_args(); rng=np.random.default_rng(args.seed); rows=[]
    for layout in range(args.n_layouts):
        payload=rng.uniform(-.02,.02,2); mass=float(rng.choice([.01,.03,.09])); friction=float(rng.choice([.2,.8,1.5])); goal=int(rng.integers(3))
        # Targets are continuous around the realized nominal endpoint, but
        # are sampled independently of the action direction.
        endpoints=[run(mass,friction,d,payload,speed=args.speed) for d in range(3)]
        target=endpoints[goal][:2]+rng.uniform(-.025,.025,2)
        wrongs=np.stack([endpoints[(goal+1)%3][:2]+rng.uniform(-.025,.025,2),endpoints[(goal+2)%3][:2]+rng.uniform(-.025,.025,2)])
        for d,p in enumerate(endpoints):
            pts=np.vstack([target,wrongs]); dist=np.linalg.norm(pts-p[:2],axis=1); nearest=int(dist.argmin())
            cls=2 if dist[nearest]<.035 and nearest==0 else (1 if dist[nearest]<.035 else 0)
            rows.append({'layout':layout,'direction':d,'mass':mass,'friction':friction,'target':target.tolist(),'payload':payload.tolist(),'class':cls})
    split=set(range(0,args.n_layouts,2)); train=[r for r in rows if r['layout'] not in split]; test=[r for r in rows if r['layout'] in split]
    out={'n_layouts':args.n_layouts,'seed':args.seed,'speed':args.speed,'n_rows':len(rows),'class_counts':{str(k):sum(r['class']==k for r in rows) for k in range(3)},'action_only':fit(train,test,'action'),'geometry_only':fit(train,test,'geometry'),'action_geometry':fit(train,test,'joint')}
    out['joint_minus_action']=out['action_geometry']['macro_auroc']-out['action_only']['macro_auroc']; Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
