"""Download a bounded file subset from a Hugging Face dataset repository."""
from __future__ import annotations
import argparse,fnmatch,json,urllib.request
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo-id',default='nvidia/BridgeData2_LeRobot_v3'); ap.add_argument('--revision',default='main'); ap.add_argument('--output',required=True); ap.add_argument('--pattern',action='append',default=['meta/info.json','meta/episodes/**']); ap.add_argument('--max-files',type=int,default=8); ap.add_argument('--max-bytes',type=int,default=2_000_000_000); ap.add_argument('--base-url',default='https://huggingface.co'); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    api=f'{a.base_url}/api/datasets/{a.repo_id}/tree/{a.revision}?recursive=true&expand=false&limit=10000'; req=urllib.request.Request(api,headers={'User-Agent':'icra2027-agcd-audit/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=60) as r: entries=json.load(r)
    except Exception as exc:
        raise SystemExit(f'Hugging Face metadata unavailable ({exc}); retry when network access or authentication is restored')
    files=[x for x in entries if x.get('type')=='file' and any(fnmatch.fnmatch(x.get('path',''),p) for p in a.pattern)]
    files=sorted(files,key=lambda x:x.get('path',''))[:a.max_files]; total=sum(int(x.get('size') or 0) for x in files)
    summary={'repo_id':a.repo_id,'revision':a.revision,'files':[{'path':x.get('path'),'size':x.get('size')} for x in files],'total_bytes':total,'max_bytes':a.max_bytes}
    print(json.dumps(summary,indent=2))
    if total>a.max_bytes: raise SystemExit('selected files exceed --max-bytes; reduce --max-files or use narrower --pattern')
    if a.dry_run: return
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    for item in files:
        rel=item['path']; dest=root/rel; dest.parent.mkdir(parents=True,exist_ok=True); url=f'{a.base_url}/datasets/{a.repo_id}/resolve/{a.revision}/{rel}?download=true'; ureq=urllib.request.Request(url,headers={'User-Agent':'icra2027-agcd-audit/1.0'})
        try:
            src_ctx=urllib.request.urlopen(ureq,timeout=120)
        except Exception as exc:
            raise SystemExit(f'cannot download {rel} ({exc}); retry when network access or authentication is restored')
        with src_ctx as src, dest.open('wb') as dst:
            written=0
            while True:
                chunk=src.read(1024*1024)
                if not chunk: break
                written+=len(chunk)
                if written>int(item.get('size') or 0)+1024*1024: dest.unlink(missing_ok=True); raise SystemExit(f'unexpected response size for {rel}')
                dst.write(chunk)
    (root/'ACQUISITION_MANIFEST.json').write_text(json.dumps(summary,indent=2)); print(f'wrote {len(files)} files to {root}')
if __name__=='__main__': main()
