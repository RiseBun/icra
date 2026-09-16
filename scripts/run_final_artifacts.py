"""Run bounded, reproducible final-artifact stages.

Heavy simulation/training remains explicit. This entry point only runs stages
whose input paths are supplied, so a missing public archive cannot create a
misleading empty result.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    return subprocess.run(cmd,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--public-input'); ap.add_argument('--output-dir',default='results/final_artifacts'); ap.add_argument('--aggregate-dir'); ap.add_argument('--skip-match',action='store_true'); ap.add_argument('--skip-stratify',action='store_true'); a=ap.parse_args()
    root=Path(__file__).resolve().parent; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); manifest={'stages':[]}
    if a.public_input:
        audit=out/'public_audit.json'; run([sys.executable,str(root/'audit_public_trajectory.py'),'--input',a.public_input,'--output',str(audit)]); manifest['stages'].append({'name':'public_audit','status':'ok','output':str(audit)})
        if not a.skip_match:
            matched=out/'public_matched_amplitude.json'; run([sys.executable,str(root/'matched_amplitude_resample.py'),'--audit',str(audit),'--output',str(matched)]); manifest['stages'].append({'name':'matched_amplitude','status':'ok','output':str(matched)})
        if not a.skip_stratify:
            stratified=out/'public_task_stratified.json'; run([sys.executable,str(root/'stratify_public_audit.py'),'--audit',str(audit),'--output',str(stratified)]); manifest['stages'].append({'name':'task_stratified_public_audit','status':'ok','output':str(stratified)})
    else:
        manifest['stages'].append({'name':'public_audit','status':'blocked','reason':'supply --public-input after acquiring a bounded archive or local export'})
    if a.aggregate_dir:
        agg=out/'aggregate.json'; run([sys.executable,str(root/'aggregate_final_results.py'),'--input-dir',a.aggregate_dir,'--output',str(agg)]); manifest['stages'].append({'name':'aggregate','status':'ok','output':str(agg)})
    manifest['repository_root']=str(root.parent); (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
