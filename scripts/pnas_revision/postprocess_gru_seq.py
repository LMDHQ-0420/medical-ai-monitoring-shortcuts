"""以训练集统计量复用 MIMIC GRU-seq 权重作年龄和 eICU 外部评估。"""
from __future__ import annotations
import argparse,json,os,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.models.gru_seq_model import GRUSequenceModel,FittedGRUSequence
from scripts.pnas_revision.postprocess_tabular import age_label,contrasts
GROUPS=('D','O','DT','F')

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def input_channels(seq,window,stats):
 raw=seq['values'][:,:window,:];cnt=seq['counts'][:,:window,:];treatment=seq['treatment'][:,:window,:]
 n,h,v=raw.shape;median=np.asarray(stats['median']);mean=np.asarray(stats['mean']);std=np.asarray(stats['std'])
 filled=raw.copy()
 for t in range(1,h):
  missing=np.isnan(filled[:,t,:]);filled[:,t,:][missing]=filled[:,t-1,:][missing]
 filled=np.where(np.isnan(filled),median[None,None,:],filled)
 d=((filled-mean)/std).astype(np.float32)
 mask=(cnt>0).astype(np.float32);count=np.log1p(cnt.astype(np.float32));gap=np.zeros_like(mask)
 for t in range(1,h):gap[:,t,:]=np.where(mask[:,t,:]>0,0,np.minimum(gap[:,t-1,:]+1,24))
 o=np.concatenate([mask,count,gap/24],axis=2).astype(np.float32)
 state=np.maximum.accumulate(treatment,axis=1).astype(np.float32)
 return {'D':d,'O':o,'DT':np.concatenate([d,state],axis=2),'F':np.concatenate([d,o,state],axis=2)}

def run(run_dir,task,window,seed):
 cell=run_dir/'formal/gru_seq'/task/f'seed{seed}'/f'{window}h';m=json.loads((cell/'manifest.json').read_text())
 if m.get('status')!='complete' or m.get('run_type')!='formal':raise ValueError('GRU-seq 正式格点未完成')
 source=ROOT/'data/processed/eicu_hourly.npz';eicu=ROOT/'data/processed'/f'eicu_{window}h.parquet'
 if not source.exists():raise FileNotFoundError(source)
 out=cell/'external_predictions.csv.gz';post=cell/'postprocess_manifest.json';input_hash=digest(source)
 if post.exists() and out.exists():
  old=json.loads(post.read_text())
  if old.get('status')=='complete' and old.get('internal_predictions_sha256')==m['predictions_sha256'] and old.get('eicu_sequence_sha256')==input_hash:
   print('VALID POSTPROCESS CACHE',cell,flush=True);return
 with np.load(source) as seq:
  ext=pd.read_parquet(eicu,columns=['stay_id','subject_id','hospital_id','age',task]);ext=ext[ext[task].notna()].copy()
  ix=pd.Series(np.arange(len(seq['stay_id'])),index=seq['stay_id']);order=ix.loc[ext.stay_id].to_numpy()
  channels=input_channels(seq,window,m['preprocessing']['hourly_numeric'])
  static=seq['static_D'].copy();mean=np.asarray(m['preprocessing']['static_mean']);std=np.asarray(m['preprocessing']['static_std'])
  static=np.nan_to_num((static-mean)/std,nan=0.0).astype(np.float32)
  pred=pd.DataFrame({'pair_id':ext.subject_id.to_numpy(),'stay_id':ext.stay_id.to_numpy(),'hospital_id':ext.hospital_id.to_numpy(),
                     'age':ext.age.to_numpy(),'y_true':ext[task].astype(int).to_numpy()})
  for group in GROUPS:
   state=torch.load(cell/f'model_{group}.pt',map_location='cpu',weights_only=True)
   model=GRUSequenceModel(state['input_size'],state['static_size'],hidden_size=64);model.load_state_dict(state['state_dict'])
   fitted=FittedGRUSequence(model,torch.device('cpu'))
   st=static if group!='O' else np.zeros((len(static),0),dtype=np.float32)
   pred[f'pred_{group}']=fitted.predict_proba(channels[group][order],st[order])
  pred['age_group']=age_label(pred.age)
  tmp=cell/'external_predictions.tmp.csv.gz';pred.to_csv(tmp,index=False,compression='gzip',float_format='%.17g');os.replace(tmp,out)
  hospital=[]
  for h,sub in pred.groupby('hospital_id'):
   c=contrasts(sub)
   if c:hospital.append({'hospital_id':h,**c,'internal_odi_raw':m['odi_raw'],'odi_shrinkage':m['odi_raw']-c['odi_raw']})
  pd.DataFrame(hospital).to_csv(cell/'external_hospital_metrics.csv',index=False)
  age=[]
  internal=pd.read_csv(cell/'predictions.csv.gz')
  tab=pd.read_parquet(ROOT/'data/processed'/f'mimic_iv_{window}h.parquet',columns=['subject_id','age']).rename(columns={'subject_id':'pair_id'})
  internal=internal.merge(tab,on='pair_id',validate='one_to_one');internal['age_group']=age_label(internal.age)
  for dataset,frame in [('mimic_iv',internal),('eicu',pred)]:
   for g,sub in frame.groupby('age_group',observed=True):
    c=contrasts(sub)
    if c:age.append({'dataset':dataset,'age_group':str(g),**c})
  pd.DataFrame(age).to_csv(cell/'age_metrics.csv',index=False)
  post.write_text(json.dumps({'status':'complete','internal_predictions_sha256':m['predictions_sha256'],'eicu_sequence_sha256':input_hash,
    'eicu_data_sha256':digest(eicu),'valid_hospitals':len(hospital),'external_rows':len(pred),
    'label_compatibility':'unresolved; external comparison exploratory'},ensure_ascii=False,indent=2)+'\n')
  print(task,window,seed,'GRU external hospitals',len(hospital),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--window',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.run_dir.resolve(),a.task,a.window,a.seed)
if __name__=='__main__':main()
