"""Download a user-selected BridgeData V2 archive without assuming a live index."""
from __future__ import annotations
import argparse,urllib.request
from pathlib import Path
DEFAULT='https://rail.eecs.berkeley.edu/datasets/bridge_release/data/'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--url',default=DEFAULT); ap.add_argument('--output',required=True); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--max-bytes',type=int,default=30_000_000_000); a=ap.parse_args(); out=Path(a.output); print({'url':a.url,'output':str(out),'max_bytes':a.max_bytes})
    if a.dry_run: return
    if a.url.endswith('/'): raise SystemExit('The official directory currently redirects/404s; pass a concrete demos*.zip, scripted*.zip, or TFDS archive URL with --url.')
    out.parent.mkdir(parents=True,exist_ok=True); req=urllib.request.Request(a.url,method='HEAD');
    with urllib.request.urlopen(req,timeout=30) as r:
        size=int(r.headers.get('Content-Length') or 0)
    if size and size>a.max_bytes: raise SystemExit(f'archive is {size} bytes; exceeds --max-bytes')
    written=0
    try:
        with urllib.request.urlopen(a.url,timeout=60) as src, out.open('wb') as dst:
            while True:
                chunk=src.read(1024*1024)
                if not chunk: break
                written += len(chunk)
                if written > a.max_bytes:
                    dst.close(); out.unlink(missing_ok=True)
                    raise SystemExit(f'download exceeded --max-bytes ({a.max_bytes})')
                dst.write(chunk)
    except Exception:
        out.unlink(missing_ok=True)
        raise
    print(f'downloaded {out} ({written} bytes)')
if __name__=='__main__': main()
