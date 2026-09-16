"""Export frozen Omega tokens for a multi-view RGB dataset."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.omega_latent import OmegaLatentConfig,OmegaLatentExtractor

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--batch-size',type=int,default=1); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args()
    src=np.load(args.input); rgb=src['rgb']; n=len(rgb) if args.limit<=0 else min(args.limit,len(rgb)); rgb=rgb[:n]
    # [N,V,H,W,3] -> [N,V,3,H,W], preserving camera order.
    frames=np.transpose(rgb,(0,1,4,2,3)); ext=OmegaLatentExtractor(OmegaLatentConfig(device=args.device,resolution=256,layer_index=4)); scene=[]; pooled=[]
    for s in range(0,n,args.batch_size):
        out=ext.encode(torch.from_numpy(frames[s:s+args.batch_size])); scene.append(out.scene_tokens.float().mean(2).half().cpu().numpy()); pooled.append(out.tokens.float().mean((1,2)).half().cpu().numpy()); print(json.dumps({'done':min(s+args.batch_size,n),'total':n}),flush=True)
    scene_np=np.concatenate(scene); pooled_np=np.concatenate(pooled); o=Path(args.output); o.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(o,scene_tokens=scene_np,pooled_tokens=pooled_np,success=src['success'][:n],direction=src['direction'][:n],target_xy=src['target_xy'][:n],payload_xy=src['payload_xy'][:n],source=np.asarray('frozen_vggt_omega_multiview'),views=np.asarray(rgb.shape[1],np.int32)); print(json.dumps({'output':str(o),'scene_tokens':scene_np.shape,'pooled_tokens':pooled_np.shape}))
if __name__=='__main__': main()
