"""Release-phase diagnostic for a physical drop class."""
import argparse,json
from pathlib import Path
import numpy as np, pybullet as pb
DIRS=np.array([[1.,0.],[-.5,.8660254],[-.5,-.8660254]])
def rollout(direction,release):
    cid=pb.connect(pb.DIRECT); pb.setGravity(0,0,-9.81); pb.setTimeStep(.01)
    plane=pb.createCollisionShape(pb.GEOM_PLANE); pb.createMultiBody(0,plane)
    ec=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.022); pc=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.018)
    ee=pb.createMultiBody(1,ec,basePosition=[0,0,.07]); pay=pb.createMultiBody(.03,pc,basePosition=[0,0,.018])
    pb.changeDynamics(ee,-1,lateralFriction=1.0); pb.changeDynamics(pay,-1,lateralFriction=.8)
    d=DIRS[direction]; start=-d*.045; pb.resetBasePositionAndOrientation(ee,[*start,.07],[0,0,0,1]); pb.resetBasePositionAndOrientation(pay,[0,0,.018],[0,0,0,1])
    for t in range(80):
        v=.30 if 5<=t<55 else 0.; z=.07 if not release or t<55 else (.07+.012*(t-55))
        pb.resetBaseVelocity(ee,[float(d[0]*v),float(d[1]*v),float(.012 if release and t>=55 else 0)],[0,0,0]); pb.stepSimulation()
    p,_=pb.getBasePositionAndOrientation(pay); pb.disconnect(cid); return np.asarray(p)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=60); ap.add_argument('--output',required=True); a=ap.parse_args(); out={}
    for rel in (False,True):
        pts=np.array([rollout(i%3,rel) for i in range(a.n)]); out[str(rel)]={'drop_rate':float(np.mean(pts[:,2]<.01)),'mean_z':float(pts[:,2].mean()),'mean_xy':float(np.linalg.norm(pts[:,:2],axis=1).mean())}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
