"""Contact-driven push stress test for external validity.

Unlike the factorial protocol, the pusher is velocity-controlled and the
payload is never teleported. Mass and friction therefore affect the endpoint.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pybullet as pb

DIRS = np.array([[1.,0.],[-.5,.8660254],[-.5,-.8660254]])

def make(mass, friction):
    cid=pb.connect(pb.DIRECT); pb.setGravity(0,0,-9.81); pb.setTimeStep(.01)
    plane=pb.createCollisionShape(pb.GEOM_PLANE); pb.createMultiBody(0,plane)
    ee_col=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.022)
    pay_col=pb.createCollisionShape(pb.GEOM_SPHERE,radius=.018)
    ee=pb.createMultiBody(1,ee_col,basePosition=[0,0,.07])
    pay=pb.createMultiBody(mass,pay_col,basePosition=[0,0,.018])
    pb.changeDynamics(ee,-1,lateralFriction=1.0,rollingFriction=.001)
    pb.changeDynamics(pay,-1,lateralFriction=friction,rollingFriction=.001)
    return cid,ee,pay

def rollout(cid,ee,pay,direction,drive=0.16):
    d=DIRS[direction]; payload=np.zeros(2); start=payload-d*.045
    pb.resetBasePositionAndOrientation(ee,[*start,.07],[0,0,0,1])
    pb.resetBasePositionAndOrientation(pay,[0,0,.018],[0,0,0,1])
    pb.resetBaseVelocity(ee,[0,0,0],[0,0,0]); pb.resetBaseVelocity(pay,[0,0,0],[0,0,0])
    # Move through the contact region with a bounded velocity; no position
    # reset is used after initialization.
    for t in range(80):
        v=0.0 if t<5 else (0.30 if t<55 else 0.0)
        pb.resetBaseVelocity(ee,[float(d[0]*v),float(d[1]*v),0],[0,0,0])
        pb.stepSimulation()
    p,_=pb.getBasePositionAndOrientation(pay)
    return np.asarray(p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--n',type=int,default=300); args=ap.parse_args()
    out={}
    for mass in (.01,.03,.09):
        for fric in (.2,.8,1.5):
            vals=[]; cid=ee=pay=None
            try:
                cid,ee,pay=make(mass,fric)
                for i in range(args.n): vals.append(rollout(cid,ee,pay,i%3))
            finally:
                if cid is not None: pb.disconnect(cid)
            arr=np.asarray(vals); disp=np.linalg.norm(arr[:,:2],axis=1)
            out[f'm{mass}_f{fric}']={'mass':mass,'friction':fric,'mean_disp':float(disp.mean()),'std_disp':float(disp.std()),'mean_z':float(arr[:,2].mean())}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
