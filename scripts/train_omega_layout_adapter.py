"""Train a small supervised adapter on frozen Omega scene tokens."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Linear(128,6))
    def forward(self,x): return self.net(x)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--omega',required=True); ap.add_argument('--output',required=True); ap.add_argument('--epochs',type=int,default=120); args=ap.parse_args()
    d=np.load(args.input); o=np.load(args.omega); n=min(len(d['seat_positions']),len(o['scene_tokens'])); y=torch.from_numpy(d['seat_positions'][:n].astype(np.float32)); z=torch.from_numpy(o['scene_tokens'][:n].astype(np.float32).mean(1))
    idx=np.random.default_rng(2027).permutation(n); cut=int(.7*n); tr,te=idx[:cut],idx[cut:]; dev='cuda' if torch.cuda.is_available() else 'cpu'; net=Adapter(z.shape[1]).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4); loss=nn.SmoothL1Loss()
    xb,yb=z[tr].to(dev),y[tr].to(dev); xt,yt=z[te].to(dev),y[te].to(dev)
    for _ in range(args.epochs): opt.zero_grad(); l=loss(net(xb),yb); l.backward(); opt.step()
    net.eval();
    with torch.no_grad(): p=net(xt).cpu().numpy()
    yy=yt.cpu().numpy(); rmse=float(np.sqrt(np.mean((p-yy)**2))); rel_y=yy.reshape(-1,3,2); rel_p=p.reshape(-1,3,2); rel=float(np.sqrt(np.mean(((rel_y-rel_y.mean(1,keepdims=True))-(rel_p-rel_p.mean(1,keepdims=True)))**2)))
    out={'n':int(n),'train':int(len(tr)),'test':int(len(te)),'device':dev,'epochs':args.epochs,'scene_token_mean_adapter_rmse_m':rmse,'relative_rmse_m':rel,'omega_frozen':True,'trainable_parameters':sum(q.numel() for q in net.parameters())}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
