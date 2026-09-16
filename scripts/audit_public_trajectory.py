"""Audit action-amplitude shortcuts on public robot trajectories.

Labels are explicitly marked as terminal_reward or progress_proxy. The proxy
uses within-episode state progress when available; it is never derived from
action magnitude.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from public_trajectory_utils import iter_episodes,amplitude

def bootstrap(y,s,groups,n=1000,seed=2027):
    rng=np.random.default_rng(seed); vals=[]; groups=np.asarray(groups); unique=np.unique(groups)
    for _ in range(n):
        sampled=rng.choice(unique,len(unique),replace=True); ix=np.concatenate([np.flatnonzero(groups==g) for g in sampled]);
        if len(np.unique(y[ix]))>1: vals.append(roc_auc_score(y[ix],s[ix]))
    return [float(x) for x in np.percentile(vals,[2.5,50,97.5])] if vals else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--max-files',type=int); ap.add_argument('--max-episodes',type=int); ap.add_argument('--history',type=int,default=8); ap.add_argument('--horizon',type=int,default=8); ap.add_argument('--quantile',type=float,default=.5); ap.add_argument('--bootstrap',type=int,default=1000); a=ap.parse_args(); rows=[]; errors=[]; episode_count=0
    # LeRobot shards can contain tens of thousands of episodes. Read the
    # tabular shard once and stream only the first requested episode groups.
    if str(a.input).lower().endswith('.parquet'):
        import pandas as pd
        frame=pd.read_parquet(a.input, columns=['action','observation.state','episode_index','task_index'])
        task_names={}
        task_file=Path(a.input).parents[2]/'meta'/'tasks.parquet'
        if task_file.exists():
            try:
                td=pd.read_parquet(task_file)
                task_names={int(v):str(k) for k,v in td['task_index'].items()}
            except Exception: task_names={}
        for ep, group in frame.groupby('episode_index', sort=True):
            if a.max_episodes is not None and episode_count >= a.max_episodes: break
            episode_count += 1
            ti=int(group['task_index'].iloc[0]); e={'actions':np.stack(group['action'].to_numpy()),'states':np.stack(group['observation.state'].to_numpy()),'meta':{'task_index':ti,'task':task_names.get(ti,f'task_{ti}')}}
            eid=f'{a.input}#episode_{int(ep)}'
            acts=e['actions']; states=e['states']
            if len(acts)<a.history+a.horizon: continue
            starts=range(0,len(acts)-a.history-a.horizon+1,a.horizon); final_state=np.asarray(states,float)[-1]
            s=np.asarray(states,float); denom=float(np.linalg.norm(s[0]-final_state))+1e-8
            for t in starts:
                end=min(t+a.history+a.horizon,len(s)-1); progress=float(1.-np.clip(np.linalg.norm(s[end]-final_state)/denom,0,1)); mean,rms,total=amplitude(acts[t:t+a.horizon]); rows.append({'episode_id':eid,'window_start':t,'mean_amp':mean,'rms_amp':rms,'total_amp':total,'progress':progress,'label_source':'progress_proxy','skill':e['meta']['task'],'task_index':e['meta']['task_index'],'environment':'unknown','source':'lerobot_parquet'})
        # Avoid the generic object-based iterator below.
        rows_ready=True
    else:
        rows_ready=False
    for eid,e in ([] if rows_ready else iter_episodes(a.input,a.max_files)):
        if a.max_episodes is not None and episode_count >= a.max_episodes: break
        episode_count += 1
        if 'error' in e: errors.append({'episode_id':eid,'error':e['error']}); continue
        acts=e['actions']; states=e.get('states');
        if len(acts)<a.history+a.horizon: continue
        starts=range(0,len(acts)-a.history-a.horizon+1,a.horizon)
        final_state=None if states is None else np.asarray(states,float)[-1]
        initial_state=None if states is None else np.asarray(states,float)[0]
        reward=e['meta'].get('reward',e['meta'].get('rewards'))
        if reward is not None:
            rr=np.asarray(reward).reshape(-1); label=int(float(rr[-1])>0.5); source='terminal_reward'
            for t in starts:
                mean,rms,total=amplitude(acts[t:t+a.horizon]); rows.append({'episode_id':eid,'window_start':t,'mean_amp':mean,'rms_amp':rms,'total_amp':total,'progress':None,'label_source':source,'label':label,'skill':e['meta'].get('skill',e['meta'].get('task','unknown')),'environment':e['meta'].get('environment',e['meta'].get('env','unknown')),'source':e['meta'].get('data_type',e['meta'].get('source','unknown'))})
        else:
            if states is None or len(states)<len(acts): continue
            s=np.asarray(states,float); denom=float(np.linalg.norm(s[0]-final_state))+1e-8
            for t in starts:
                end=min(t+a.history+a.horizon,len(s)-1); progress=float(1.-np.clip(np.linalg.norm(s[end]-final_state)/denom,0,1)); mean,rms,total=amplitude(acts[t:t+a.horizon]); rows.append({'episode_id':eid,'window_start':t,'mean_amp':mean,'rms_amp':rms,'total_amp':total,'progress':progress,'label_source':'progress_proxy','skill':e['meta'].get('skill',e['meta'].get('task','unknown')),'environment':e['meta'].get('environment',e['meta'].get('env','unknown')),'source':e['meta'].get('data_type',e['meta'].get('source','unknown'))})
    if not rows: raise SystemExit('no analyzable episodes')
    proxy=[r for r in rows if r.get('label_source')=='progress_proxy']
    if proxy:
        threshold=float(np.median([r['progress'] for r in proxy]))
        for r in proxy: r['label']=int(r['progress']>=threshold); r['label_threshold']=threshold
    y=np.asarray([int(r['label']) for r in rows]); score=np.asarray([r['mean_amp'] for r in rows]);
    if len(np.unique(y))<2: result={'status':'no_label_variation','n_windows':len(rows),'n_episodes':len(set(r['episode_id'] for r in rows))}
    else:
        x=score[:,None]; tr=np.arange(len(y)); clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(x,y); p=clf.predict_proba(x)[:,1]
        for r in rows: r['action_only_score']=None
        groups=np.asarray([r['episode_id'] for r in rows]); unique=np.unique(groups); rng=np.random.default_rng(2027); perm=rng.permutation(unique); cut=max(1,int(.7*len(unique))); trset=set(perm[:cut]); teix=np.asarray([g not in trset for g in groups]);
        if teix.sum() and len(np.unique(y[teix]))>1:
            test_clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(x[~teix],y[~teix]); p_test=test_clf.predict_proba(x[teix])[:,1]; test_auc=float(roc_auc_score(y[teix],p_test)); test_brier=float(brier_score_loss(y[teix],p_test)); test_auprc=float(average_precision_score(y[teix],p_test));
            for r,pi in zip([r for r,k in zip(rows,teix) if k],p_test): r['action_only_score']=float(pi)
        else: test_auc=test_brier=test_auprc=None
        result={'status':'ok','n_windows':len(rows),'n_episodes':len(set(r['episode_id'] for r in rows)),'positive_rate':float(y.mean()),'amplitude_auroc':float(roc_auc_score(y,score)),'amplitude_auroc_ci95':bootstrap(y,score,groups,a.bootstrap),'action_only_auroc':test_auc,'action_only_brier':test_brier,'action_only_auprc':test_auprc,'action_only_split':'episode_disjoint_70_30','label_sources':sorted(set(r['label_source'] for r in rows)),'history':a.history,'horizon':a.horizon,'proxy_threshold':next((r.get('label_threshold') for r in rows if 'label_threshold' in r),None)}
    result.update({'dataset':'public_trajectory','errors':errors[:50],'episode_ids_hash':hashlib.sha256('|'.join(r['episode_id'] for r in rows).encode()).hexdigest(),'rows':rows})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
