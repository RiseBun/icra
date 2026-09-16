"""Three-class action x geometry factor test with physical carry/release."""
import argparse,json
from pathlib import Path
import numpy as np, pybullet as pb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
DIRS=np.array([[1.,0.],[-.5,.8660254],[-.5,-.8660254]])
def rollout(d, release_step):
    cid=pb.connect(pb.DIRECT); pb.setGravity(0,0,-9.81); pb.setTimeStep(.01)
    plane=pb.createCollisionShape(pb.GEOM_PLANE); pb.createMultiBody(0,plane)
    ec=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.022); pc=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.018)
    ee=pb.createMultiBody(1,ec,basePosition=[0,0,.14]); pay=pb.createMultiBody(.03,pc,basePosition=[0,0,.018]); pb.changeDynamics(pay,-1,lateralFriction=.8,restitution=.05)
    v=DIRS[d]; start=-v*.045; pb.resetBasePositionAndOrientation(ee,[*start,.14],[0,0,0,1]); pb.resetBasePositionAndOrientation(pay,[*start,.10],[0,0,0,1])
    for t in range(90):
        a=min(1.,(t+1)/55); pos=start*(1-a)+(start+v*.16)*a; pb.resetBasePositionAndOrientation(ee,[*pos,.14],[0,0,0,1])
        if t<release_step: pb.resetBasePositionAndOrientation(pay,[*pos,.10],[0,0,0,1]); pb.resetBaseVelocity(pay,[0,0,0],[0,0,0])
        pb.stepSimulation()
    p,_=pb.getBasePositionAndOrientation(pay); pb.disconnect(cid); return np.asarray(p)
def fit(train,test,mode):
    def f(r):
        a=np.eye(3)[r['direction']]
        return np.r_[a] if mode=='action' else np.r_[a,r['target'],r['release']] if mode=='joint' else np.r_[r['target'],r['release']]
    x=np.stack([f(r) for r in train]); xt=np.stack([f(r) for r in test]); y=np.array([r['class'] for r in train]); yt=np.array([r['class'] for r in test]); c=MLPClassifier(hidden_layer_sizes=(48,32),max_iter=900,random_state=2027).fit(x,y); p=c.predict_proba(xt); pred=c.classes_[p.argmax(1)]; vals=[]
    for j,k in enumerate(c.classes_):
        z=(yt==k).astype(int)
        if len(np.unique(z))==2: vals.append(roc_auc_score(z,p[:,j]))
    return {'macro_auroc':float(np.mean(vals)),'accuracy':float(accuracy_score(yt,pred))}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n-layouts',type=int,default=60); ap.add_argument('--output',required=True); ap.add_argument('--seed',type=int,default=2027); a=ap.parse_args(); rng=np.random.default_rng(a.seed); rows=[]
    for lay in range(a.n_layouts):
        goal=int(rng.integers(3)); centers=DIRS*.115+rng.uniform(-.012,.012,(3,2)); target=centers[goal];
        for d in range(3):
            rel=20 if (d==((goal+1)%3) and lay%2==0) else 45; p=rollout(d,rel); dist=np.linalg.norm(p[:2]-target); wrong=np.min(np.linalg.norm(centers[np.arange(3)!=goal]-p[:2],axis=1)); cls=0 if rel<40 or p[2]>.04 else (2 if dist<.035 else (1 if wrong<.035 else 0)); rows.append({'layout':lay,'direction':d,'target':target.tolist(),'release':rel/45.,'class':cls})
    split=set(range(0,a.n_layouts,2)); tr=[r for r in rows if r['layout'] not in split]; te=[r for r in rows if r['layout'] in split]; out={'n_layouts':a.n_layouts,'seed':a.seed,'n_rows':len(rows),'class_counts':{str(k):sum(r['class']==k for r in rows) for k in range(3)},'action_only':fit(tr,te,'action'),'geometry_only':fit(tr,te,'geometry'),'action_geometry':fit(tr,te,'joint')}; out['joint_minus_action']=out['action_geometry']['macro_auroc']-out['action_only']['macro_auroc']; Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
