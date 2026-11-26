"""用真实逐小时事件训练 GRU-seq；四个输入视野共享患者与填补统计量。"""
from __future__ import annotations
import hashlib,json,os,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
from src.data.transforms import split_cohort
from src.models.gru_seq_model import fit_gru_sequence
TASKS=('icu_mortality','inhosp_mortality','aki','sepsis_deterioration','escalation')
WINDOWS=(3,6,12,18,24);SEEDS=(42,43,44,45,46)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def channels(values,counts,treatment,train_idx,window):
    raw=values[:,:window,:]
    cnt=counts[:,:window,:]
    n,h,v=raw.shape
    med=np.nanmedian(raw[train_idx].reshape(-1,v),axis=0)
    med=np.nan_to_num(med,nan=0.0)
    filled=raw.copy()
    for t in range(1,h):
        missing=np.isnan(filled[:,t,:]);filled[:,t,:][missing]=filled[:,t-1,:][missing]
    filled=np.where(np.isnan(filled),med[None,None,:],filled)
    mean=filled[train_idx].mean(axis=(0,1));std=filled[train_idx].std(axis=(0,1));std=np.where(std<1e-6,1,std)
    d=((filled-mean)/std).astype(np.float32)
    mask=(cnt>0).astype(np.float32)
    count=np.log1p(cnt.astype(np.float32))
    gap=np.zeros_like(mask)
    for t in range(1,h):gap[:,t,:]=np.where(mask[:,t,:]>0,0,np.minimum(gap[:,t-1,:]+1,24))
    o=np.concatenate([mask,count,gap/24],axis=2).astype(np.float32)
    treatment_state=np.maximum.accumulate(treatment[:,:window,:],axis=1).astype(np.float32)
    return ({'D':d,'O':o,'DT':np.concatenate([d,treatment_state],axis=2),
             'F':np.concatenate([d,o,treatment_state],axis=2)},
            {'median':med.tolist(),'mean':mean.tolist(),'std':std.tolist()})

def run(run_dir,task,window,seed):
    cfg_path=ROOT/'configs/default.yaml'
    cfg=yaml.safe_load(cfg_path.read_text())['training']['gru_seq']
    cfg_hash=sha(cfg_path)
    source=ROOT/'data/processed/mimic_iv_hourly.npz'
    if not source.exists():raise FileNotFoundError(source)
    source_hash=sha(source)
    data=ROOT/'data/processed'/f'mimic_iv_{window}h.parquet'
    df=pd.read_parquet(data,columns=['stay_id','subject_id','age','gender',task]).loc[lambda x:x[task].notna()].copy()
    train,val,test=split_cohort(df,seed=seed)
    seq=np.load(source)
    idx=pd.Series(np.arange(len(seq['stay_id'])),index=seq['stay_id'])
    if not df.stay_id.isin(idx.index).all():raise ValueError('序列与表格队列 stay 键不匹配')
    order=idx.loc[df.stay_id].to_numpy()
    train_idx=idx.loc[train.stay_id].to_numpy();val_idx=idx.loc[val.stay_id].to_numpy();test_idx=idx.loc[test.stay_id].to_numpy()
    all_channels,stats=channels(seq['values'],seq['counts'],seq['treatment'],train_idx,window)
    static=seq['static_D'].copy()
    smean=static[train_idx].mean(axis=0);sstd=static[train_idx].std(axis=0);sstd=np.where(sstd<1e-6,1,sstd)
    static=((static-smean)/sstd).astype(np.float32)
    out=run_dir/'models/gru_seq'/task/f'seed{seed}'/f'{window}h';out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'manifest.json';pred_path=out/'predictions.csv.gz'
    if manifest_path.exists() and pred_path.exists():
        prev=json.loads(manifest_path.read_text())
        if prev.get('status')=='complete' and prev.get('config_sha256')==cfg_hash and prev.get('sequence_sha256')==source_hash and prev.get('data_sha256')==sha(data) and prev.get('predictions_sha256')==sha(pred_path):
            print('VALID CACHE',out,flush=True);return
    pred=pd.DataFrame({'pair_id':test.subject_id.to_numpy(),'stay_id':test.stay_id.to_numpy(),'y_true':test[task].astype(int).to_numpy()})
    m={'status':'running','run_type':'formal','model':'gru_seq','task':task,'window_h':window,'seed':seed,'sequence_sha256':source_hash,'data_sha256':sha(data),'config_sha256':cfg_hash,
       'split_id':hashlib.sha256('\n'.join(map(str,sorted(test.subject_id))).encode()).hexdigest(),
       'n_train':len(train),'n_val':len(val),'n_test':len(test),'positive_test':int(test[task].sum()),
       'run_id':json.loads((run_dir/'analysis_protocol.json').read_text()).get('protocol_id','formal'),'static_D':['age','gender'],'D_channels':len(stats['median']),'O_channels':len(stats['median'])*3,'T_channels':3,
       'time_steps':window,'hyperparameters':cfg,'failure_reason':None}
    manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    try:
        for group,x in all_channels.items():
            st=static if group!='O' else np.zeros((len(static),0),dtype=np.float32)
            model,fit=fit_gru_sequence(x[train_idx],st[train_idx],train[task].to_numpy(),
                x[val_idx],st[val_idx],val[task].to_numpy(),seed=seed,hidden_size=cfg['hidden_size'],max_epochs=cfg['max_epochs'],patience=cfg['patience'],batch_size=cfg['batch_size'],learning_rate=cfg['lr'])
            score=model.predict_proba(x[test_idx],st[test_idx]);pred[f'pred_{group}']=score
            m[f'auc_{group}']=float(roc_auc_score(pred.y_true,score));m[f'fit_{group}']=fit
            torch.save({'state_dict':model.model.cpu().state_dict(),'input_size':x.shape[2],'static_size':st.shape[1]},out/f'model_{group}.pt')
            print(task,window,seed,group,m[f'auc_{group}'],flush=True)
        m['odi_raw']=m['auc_F']-m['auc_D'];m['odi_residual']=m['auc_F']-m['auc_DT'];m['reference_gain']=m['auc_DT']-m['auc_D']
        if abs(m['odi_raw']-m['odi_residual']-m['reference_gain'])>1e-12:raise ValueError('ODI 恒等式失败')
        tmp=out/'predictions.tmp.csv.gz';pred.to_csv(tmp,index=False,compression='gzip',float_format='%.17g');os.replace(tmp,pred_path)
        m['predictions_sha256']=sha(pred_path);m['preprocessing']={'hourly_numeric':stats,'static_mean':smean.tolist(),'static_std':sstd.tolist()}
        m['status']='complete'
    except Exception as exc:
        m['status']='failed';m['failure_reason']=repr(exc);m['traceback']=traceback.format_exc();raise
    finally:manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
