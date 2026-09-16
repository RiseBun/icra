"""Small, dependency-light readers and features for public robot trajectories."""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np

def _first(d, keys):
    for k in keys:
        if k in d: return d[k]
    return None

def _json_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value

def _numeric_array(value):
    a=np.asarray(value)
    if a.dtype == object and a.ndim == 1 and len(a):
        try: a=np.stack(a)
        except ValueError: pass
    return a

def load_trajectory(path: Path):
    suf=path.suffix.lower()
    if suf=='.npz':
        z=np.load(path,allow_pickle=True); return {k:z[k] for k in z.files}
    if suf in ('.json','.jsonl'):
        text=path.read_text(encoding='utf-8');
        if suf=='.jsonl': return {'rows':[json.loads(x) for x in text.splitlines() if x.strip()]}
        return json.loads(text)
    if suf in ('.pkl','.pickle'):
        with path.open('rb') as f: return pickle.load(f)
    if suf in ('.h5','.hdf5'):
        import h5py
        with h5py.File(path,'r') as f:
            keys=list(f.keys());
            if len(keys)==1 and hasattr(f[keys[0]],'keys'): g=f[keys[0]]
            else: g=f
            return {k:np.asarray(g[k]) for k in g.keys() if hasattr(g[k],'shape')}
    if suf=='.parquet':
        import pandas as pd
        frame=pd.read_parquet(path)
        # Keep column names and grouping information for iter_episodes.
        return {k:frame[k].to_numpy() for k in frame.columns}
    raise ValueError(f'unsupported trajectory format: {path}')

def normalize_episode(obj):
    if 'rows' in obj:
        rows=obj['rows']; return {'actions':np.asarray([r.get('action',r.get('actions')) for r in rows]), 'states':np.asarray([r.get('state',r.get('states')) for r in rows]), 'images':None, 'meta':{}}
    observations=_first(obj,['observations','observation'])
    if not isinstance(observations,dict): observations=None
    actions=_first(obj,['actions','action','action_array','action_dict'])
    states=_first(obj,['states','state','proprio','observations/state'])
    images=_first(obj,['images','image','rgb','observations/image','observation'])
    if actions is None and observations is not None:
        actions=_first(observations,['actions','action','action_array'])
    if states is None and observations is not None:
        states=_first(observations,['states','state','proprio','cartesian_position','joint_position'])
    if images is None and observations is not None:
        images=_first(observations,['images','image','rgb','image_0','exterior_image_1_left','wrist_image_left'])
    if isinstance(actions,dict): actions=_first(actions,['cartesian_velocity','joint_velocity','action','joint_position'])
    if isinstance(states,dict): states=_first(states,['cartesian_position','joint_position','state'])
    if isinstance(images,dict): images=_first(images,['image_0','exterior_image_1_left','wrist_image_left','rgb'])
    if actions is None: raise ValueError('trajectory has no actions')
    a=_numeric_array(actions); s=None if states is None else _numeric_array(states)
    if a.ndim==1: a=a[:,None]
    excluded=('actions','action','action_array','action_dict','states','state','proprio','images','image','rgb','observations','observation')
    meta={k:_json_value(v) for k,v in obj.items() if k not in excluded}
    return {'actions':a,'states':s,'images':None if images is None else np.asarray(images),'meta':meta}

def iter_episodes(root, max_files=None):
    root=Path(root); files=[]
    if root.is_file(): files=[root]
    else:
        for pat in ('*.npz','*.json','*.jsonl','*.pkl','*.pickle','*.h5','*.hdf5','*.parquet'): files.extend(root.rglob(pat))
    for p in sorted(files)[:max_files or None]:
        try:
            obj=load_trajectory(p)
            if p.suffix.lower()=='.parquet' and 'episode_index' in obj:
                groups={int(x) for x in np.unique(obj['episode_index'])}
                for ep in sorted(groups):
                    mask=np.asarray(obj['episode_index'])==ep
                    actions=_first(obj,['action','actions']); states=_first(obj,['observation.state','state','states'])
                    if actions is None: raise ValueError('parquet trajectory has no action column')
                    item={'actions':_numeric_array(actions)[mask], 'states':None if states is None else _numeric_array(states)[mask], 'episode_index':ep}
                    yield f'{p}#episode_{ep}', normalize_episode(item)
            elif isinstance(obj,dict) and 'episodes' in obj:
                for i,e in enumerate(obj['episodes']): yield str(p)+f'#{i}', normalize_episode(e)
            else: yield str(p),normalize_episode(obj)
        except Exception as exc:
            yield str(p),{'error':str(exc)}

def amplitude(actions):
    a=np.asarray(actions,float)
    if len(a)<2: return 0.,0.,0.
    step=np.linalg.norm(a,axis=1); return float(step.mean()),float(np.sqrt((step**2).mean())),float(step.sum())

def phase(actions, n=4):
    t=len(actions); return min(n-1,int(np.floor(np.arange(t)[-1]*n/max(t,1))))

def state_endpoint_progress(states):
    if states is None or len(states)<2: return None
    s=np.asarray(states,float); d=np.linalg.norm(s-s[-1],axis=1); scale=np.percentile(d,95)+1e-8
    return float(1.-np.clip(d[0]/scale,0,1)),float(np.clip(1.-d[0]/scale,0,1))
