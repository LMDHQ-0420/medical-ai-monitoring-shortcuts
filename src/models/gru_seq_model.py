"""真实逐小时序列 GRU；静态临床变量在编码后拼接。"""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

class GRUSequenceModel(nn.Module):
    def __init__(self,input_size:int,static_size:int=0,hidden_size:int=64):
        super().__init__()
        self.gru=nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=1,batch_first=True,bidirectional=False)
        self.head=nn.Linear(hidden_size+static_size,1)
    def forward(self,sequence,static=None):
        if sequence.ndim!=3 or sequence.shape[1]<2:
            raise ValueError('GRU-seq 需要至少两个真实时间步的 [batch,time,features] 输入')
        _,state=self.gru(sequence)
        hidden=state[-1]
        if static is not None:hidden=torch.cat([hidden,static],dim=1)
        return torch.sigmoid(self.head(hidden)).squeeze(-1)

class FittedGRUSequence:
    def __init__(self,model,device):self.model=model;self.device=device
    def predict_proba(self,sequence,static=None,batch_size=1024):
        sequence=np.asarray(sequence,dtype=np.float32)
        if sequence.ndim!=3 or sequence.shape[1]<2:raise ValueError('GRU-seq 输入必须有真实的多个时间步')
        if static is None:static=np.zeros((len(sequence),0),dtype=np.float32)
        out=[];self.model.eval()
        with torch.no_grad():
            for start in range(0,len(sequence),batch_size):
                x=torch.as_tensor(sequence[start:start+batch_size],device=self.device)
                s=torch.as_tensor(static[start:start+batch_size],dtype=torch.float32,device=self.device)
                out.append(self.model(x,s).cpu().numpy())
        return np.concatenate(out)

def fit_gru_sequence(x_train,s_train,y_train,x_val,s_val,y_val,seed=42,hidden_size=64,
                     max_epochs=100,patience=10,batch_size=256,learning_rate=0.001):
    x_train=np.asarray(x_train,dtype=np.float32);x_val=np.asarray(x_val,dtype=np.float32)
    if x_train.ndim!=3 or x_train.shape[1]<2 or x_val.ndim!=3 or x_val.shape[1]!=x_train.shape[1]:
        raise ValueError('训练及验证输入必须是相同真实时间格的三维序列')
    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():raise ValueError('输入包含未填补的非有限值')
    torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=GRUSequenceModel(x_train.shape[2],s_train.shape[1],hidden_size).to(device)
    tr=TensorDataset(torch.from_numpy(x_train),torch.from_numpy(np.asarray(s_train,dtype=np.float32)),torch.from_numpy(np.asarray(y_train,dtype=np.float32)))
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(tr,batch_size=batch_size,shuffle=True,generator=gen)
    optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_fn=nn.BCELoss()
    xv=torch.as_tensor(x_val,device=device);sv=torch.as_tensor(s_val,dtype=torch.float32,device=device);yv=torch.as_tensor(y_val,dtype=torch.float32,device=device)
    best=float('inf');best_state=None;stale=0
    for epoch in range(max_epochs):
        model.train()
        for xb,sb,yb in loader:
            xb,sb,yb=xb.to(device),sb.to(device),yb.to(device)
            optim.zero_grad();loss=loss_fn(model(xb,sb),yb);loss.backward();optim.step()
        model.eval()
        with torch.no_grad():val_loss=float(loss_fn(model(xv,sv),yv))
        if val_loss<best-1e-6:best=val_loss;best_state=copy.deepcopy(model.state_dict());stale=0
        else:
            stale+=1
            if stale>=patience:break
    if best_state is not None:model.load_state_dict(best_state)
    return FittedGRUSequence(model,device),{'best_val_loss':best,'epochs':epoch+1,'parameters':sum(p.numel() for p in model.parameters())}
