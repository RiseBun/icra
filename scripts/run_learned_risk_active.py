"""Closed-loop learned risk selection with an optional second view."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

def features(seats,target_xy,payload,direction):
    return np.r_[np.eye(3)[int(direction)], .16, target_xy, seats.reshape(-1), payload]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--interaction",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
    rows=json.loads(Path(args.interaction).read_text()); lab={"drop":0,"wrong_target":1,"success":2}; y=np.array([lab[r["label"]] for r in rows]); idx=np.random.default_rng(2027).permutation(len(y)); cut=len(y)//2; tr,te=idx[:cut],idx[cut:]
    Xtr=np.stack([features(np.asarray(r["seat_positions"]).reshape(3,2),r["target_xy"],r["payload_xy"],r["direction"]) for r in rows])[tr]
    clf=HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=31,random_state=0).fit(Xtr,y[tr])
    stats={"fixed_view":[],"risk_active":[]}; counts={"fixed_view":[],"risk_active":[]}
    rng=np.random.default_rng(77)
    for i in te:
        r=rows[i]; seats=np.asarray(r["seat_positions"]).reshape(3,2); target=np.asarray(r["target_xy"]); payload=np.asarray(r["payload_xy"])
        def choose(sigma):
            so=seats+rng.normal(0,sigma,seats.shape); to=target+rng.normal(0,sigma*1.5,2)
            batch=np.stack([features(so,to,payload,d) for d in range(3)])
            probs=clf.predict_proba(batch)
            # Risk prioritizes failure and drop probabilities.
            risk=probs[:,0]+probs[:,1]; d=int(np.argmin(risk)); return d,float(risk[d]),probs
        d0,r0,_=choose(.085); stats["fixed_view"].append(int(d0==r["target"])); counts["fixed_view"].append(0)
        d1,r1,_=choose(.008); use=bool(r0>.45); stats["risk_active"].append(int((d1 if use else d0)==r["target"])); counts["risk_active"].append(int(use))
    fixed=np.asarray(stats["fixed_view"],dtype=float); active=np.asarray(stats["risk_active"],dtype=float)
    fixed_obs=np.asarray(counts["fixed_view"],dtype=float); active_obs=np.asarray(counts["risk_active"],dtype=float)
    rng_boot=np.random.default_rng(2028); boot=np.empty((5000,2),dtype=float)
    for b in range(len(boot)):
        j=rng_boot.integers(0,len(te),len(te))
        boot[b,0]=np.mean(active[j]-fixed[j])
        boot[b,1]=np.mean(active[j]-.1*active_obs[j])-np.mean(fixed[j]-.1*fixed_obs[j])
    result={"n_test":len(te),"success_rate":{"fixed_view":float(np.mean(fixed)),"risk_active":float(np.mean(active))},"mean_observations":{"fixed_view":float(np.mean(fixed_obs)),"risk_active":float(np.mean(active_obs))},"utility_lambda_0.10":{"fixed_view":float(np.mean(fixed)-.1*np.mean(fixed_obs)),"risk_active":float(np.mean(active)-.1*np.mean(active_obs))},"bootstrap_95ci":{"success_gain_active_minus_fixed":[float(np.quantile(boot[:,0],.025)),float(np.quantile(boot[:,0],.975))],"utility_gain_active_minus_fixed":[float(np.quantile(boot[:,1],.025)),float(np.quantile(boot[:,1],.975))]},"threshold":.45,"view_noise_m":{"first":.085,"second":.008}}
    Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
