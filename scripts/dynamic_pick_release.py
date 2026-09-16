"""Minimal physical carry-release protocol producing success/wrong/drop."""
import argparse,json
from pathlib import Path
import numpy as np, pybullet as pb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
DIRS=np.array([[1.,0.],[-.5,.8660254],[-.5,-.8660254]])
def rollout(direction,release_step=45):
    cid=pb.connect(pb.DIRECT); pb.setGravity(0,0,-9.81); pb.setTimeStep(.01)
    plane=pb.createCollisionShape(pb.GEOM_PLANE); pb.createMultiBody(0,plane)
    ec=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.022); pc=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.018)
    ee=pb.createMultiBody(1,ec,basePosition=[0,0,.14]); pay=pb.createMultiBody(.03,pc,basePosition=[0,0,.018])
    pb.changeDynamics(pay,-1,lateralFriction=.8,restitution=.05)
    d=DIRS[direction]; start=-d*.045; pb.resetBasePositionAndOrientation(ee,[*start,.14],[0,0,0,1]); pb.resetBasePositionAndOrientation(pay,[*start,.10],[0,0,0,1])
    for t in range(90):
        a=min(1.,(t+1)/55); pos=start*(1-a)+(start+d*.16)*a; pb.resetBasePositionAndOrientation(ee,[*pos,.14],[0,0,0,1])
        if t<release_step: pb.resetBasePositionAndOrientation(pay,[*pos,.10],[0,0,0,1]); pb.resetBaseVelocity(pay,[0,0,0],[0,0,0])
        pb.stepSimulation()
    p,_=pb.getBasePositionAndOrientation(pay); pb.disconnect(cid); return np.asarray(p)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=120); ap.add_argument('--output',required=True); a=ap.parse_args(); rng=np.random.default_rng(2027); rows=[]
    for i in range(a.n):
        d=i%3; p=rollout(d,release_step=45 if i%4 else 20); target=DIRS[d]*.115+rng.uniform(-.018,.018,2); wrong=DIRS[(d+1)%3]*.115; dist=np.linalg.norm(p[:2]-target); wrongdist=np.linalg.norm(p[:2]-wrong); cls=2 if dist<.035 and p[2]<.04 else (1 if wrongdist<.035 and p[2]<.04 else 0)
        rows.append({'direction':d,'target':target.tolist(),'endpoint':p.tolist(),'class':cls})
    print(json.dumps({'n':len(rows),'counts':{str(k):sum(r['class']==k for r in rows) for k in range(3)},'mean_z':float(np.mean([r['endpoint'][2] for r in rows]))},indent=2)); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(rows,indent=2))
if __name__=='__main__': main()
