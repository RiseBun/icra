"""Aggregate final dynamic seeds and public-audit JSONs into one manifest."""
import argparse,json
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output',required=True); a=ap.parse_args(); files=sorted(Path(a.input_dir).glob('*.json')); data=[json.loads(p.read_text()) for p in files]; gains=[d['joint_minus_action'] for d in data if 'joint_minus_action' in d]; rng=np.random.default_rng(7); boots=[float(np.mean(rng.choice(gains,len(gains),replace=True))) for _ in range(20000)] if gains else []
    out={'n_files':len(files),'files':[p.name for p in files],'joint_minus_action_mean':float(np.mean(gains)) if gains else None,'joint_minus_action_ci95':[float(x) for x in np.percentile(boots,[2.5,97.5])] if boots else None,'records':data}; Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
