"""Train a compact multi-view RGB layout readout."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch import nn

class Net(nn.Module):
    def __init__(self,c):
        super().__init__(); self.v=int(c//3); self.f=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,5,2,2),nn.ReLU(),nn.Conv2d(64,128,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.camera_pose=nn.Parameter(torch.tensor([[0.,-0.68,.62],[-.62,-.35,.55],[.62,-.35,.55]],dtype=torch.float32)[:self.v]); self.h=nn.Sequential(nn.Linear(self.v*128+self.v*3,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,6))
    def forward(self,x):
        b=x.shape[0]; z=x.reshape(b,self.v,3,x.shape[-2],x.shape[-1]).flatten(0,1); z=self.f(z).reshape(b,self.v,128); pose=self.camera_pose.unsqueeze(0).expand(b,-1,-1); return self.h(torch.cat([z.reshape(b,-1),pose.reshape(b,-1)],1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--views',type=int,default=3); ap.add_argument('--epochs',type=int,default=80); args=ap.parse_args()
    d=np.load(args.input); rgb=d['rgb'][:,:args.views].astype(np.float32)/255.; x=torch.from_numpy(rgb.transpose(0,1,4,2,3).reshape(len(rgb),args.views*3,rgb.shape[2],rgb.shape[3])); y=torch.from_numpy(d['seat_positions'].astype(np.float32)); idx=np.random.default_rng(2027).permutation(len(x)); cut=int(.7*len(x)); tr,te=idx[:cut],idx[cut:]; dev='cuda' if torch.cuda.is_available() else 'cpu'; net=Net(args.views*3).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4); loss=nn.SmoothL1Loss(); xb,yb=x[tr].to(dev),y[tr].to(dev)
    for _ in range(args.epochs): opt.zero_grad(); l=loss(net(xb),yb); l.backward(); opt.step()
    net.eval();
    with torch.no_grad(): pred=net(x[te].to(dev)).cpu().numpy()
    yy=y[te].numpy(); rmse=float(np.sqrt(np.mean((pred-yy)**2))); rel_y=yy.reshape(-1,3,2); rel_p=pred.reshape(-1,3,2); rel=float(np.sqrt(np.mean(((rel_y-rel_y.mean(1,keepdims=True))-(rel_p-rel_p.mean(1,keepdims=True)))**2)))
    out={'n':int(len(x)),'views':args.views,'train':int(len(tr)),'test':int(len(te)),'device':dev,'epochs':args.epochs,'seat_rmse_m':rmse,'relative_seat_rmse_m':rel}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
