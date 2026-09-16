"""Small RGB layout readout baseline for the MuJoCo rendered scenes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch import nn

class LayoutNet(nn.Module):
    def __init__(self):
        super().__init__(); self.body=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,5,2,2),nn.ReLU(),nn.Conv2d(64,128,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,6))
    def forward(self,x): return self.head(self.body(x))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--epochs',type=int,default=40); args=ap.parse_args()
    d=np.load(args.input); x=torch.from_numpy(np.transpose(d['rgb'].astype(np.float32)/255.,(0,3,1,2))); y=torch.from_numpy(d['seat_positions'].astype(np.float32))
    idx=np.random.default_rng(2027).permutation(len(x)); cut=int(.7*len(x)); tr,te=idx[:cut],idx[cut:]
    dev='cuda' if torch.cuda.is_available() else 'cpu'; net=LayoutNet().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4); loss=nn.SmoothL1Loss()
    xb,yb=x[tr].to(dev),y[tr].to(dev); xt,yt=x[te].to(dev),y[te].to(dev)
    for ep in range(args.epochs):
        net.train(); opt.zero_grad(); l=loss(net(xb),yb); l.backward(); opt.step()
    net.eval()
    with torch.no_grad(): pred=net(xt).cpu().numpy()
    rmse=float(np.sqrt(np.mean((pred-yt.cpu().numpy())**2))); rel_y=yt.cpu().numpy().reshape(-1,3,2); rel_p=pred.reshape(-1,3,2); rel_rmse=float(np.sqrt(np.mean(((rel_y-rel_y.mean(1,keepdims=True))-(rel_p-rel_p.mean(1,keepdims=True)))**2)))
    out={'n':int(len(x)),'train':int(len(tr)),'test':int(len(te)),'device':dev,'epochs':args.epochs,'seat_rmse_m':rmse,'relative_seat_rmse_m':rel_rmse}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
