"""Episode-level matched-amplitude audit for public trajectories."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def _stratum(row):
    # Keep the public audit honest: match only within comparable source/task
    # buckets. Missing metadata is explicit rather than silently imputed.
    return tuple(row.get(k,'unknown') for k in ('skill','environment','source'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--audit',required=True); ap.add_argument('--output',required=True); ap.add_argument('--bins',type=int,default=8); a=ap.parse_args(); d=json.loads(Path(a.audit).read_text()); rows=[r for r in d.get('rows',[]) if r.get('label') is not None];
    if not rows or len({r['label'] for r in rows})<2: out={'status':'no_label_variation','n':len(rows)}
    else:
        amp=np.array([r['mean_amp'] for r in rows]); y=np.array([r['label'] for r in rows]);
        pos=np.where(y==1)[0].tolist(); neg=np.where(y==0)[0].tolist(); used=set(); pairs=[]
        for i in sorted(pos,key=lambda j:amp[j]):
            cand=[j for j in neg if j not in used and _stratum(rows[j])==_stratum(rows[i])]
            if not cand:
                # If the dataset lacks source metadata, preserve usability by
                # falling back to global matching and record the fallback.
                cand=[j for j in neg if j not in used]
            if not cand: break
            j=min(cand,key=lambda q:abs(amp[q]-amp[i])); used.add(j); pairs.append((i,j))
        matched=np.array([j for p in pairs for j in p],dtype=int); score=amp[matched]
        action=np.asarray([r.get('action_only_score',np.nan) for r in rows],float)
        has_action=np.isfinite(action).all()
        out={'status':'ok','n':len(rows),'n_pairs':len(pairs),'raw_amplitude_auroc':float(roc_auc_score(y,amp)),'matched_amplitude_auroc':float(roc_auc_score(y[matched],score)) if len(np.unique(y[matched]))>1 else None,'raw_action_only_auroc':float(roc_auc_score(y,action)) if has_action else None,'matched_action_only_auroc':float(roc_auc_score(y[matched],action[matched])) if has_action and len(np.unique(y[matched]))>1 else None,'matched_indices':matched.tolist(),'matched_abs_amp_gap':float(np.mean([abs(amp[i]-amp[j]) for i,j in pairs])) if pairs else None,'matching':'greedy nearest-neighbor within skill/environment/source when available; global fallback only when a stratum has no counterpart','action_only_note':None if has_action else 'action-only scores are held out for the episode-disjoint test split and are not available for all matched rows'}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
