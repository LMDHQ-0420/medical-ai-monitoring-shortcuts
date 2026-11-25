"""在同一患者重采样索引上复算三项 AUROC 差值的分层百分位区间。"""
from __future__ import annotations
import argparse,itertools,json,sys,time,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pnas_revision.run_tabular import TASKS,MODELS,WINDOWS,SEEDS,GROUPS,digest_file

def auc_fast(y,score):
 npos=int(y.sum());nneg=len(y)-npos
 ranks=rankdata(score,method='average')
 return float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def one(job):
 run,kind,task,model,window,seed,n_boot=job
 if kind=='tabular':cell=run/'formal/tabular'/task/model/f'seed{seed}'/f'{window}h'
 else:cell=run/'formal/gru_seq'/task/f'seed{seed}'/f'{window}h'
 p=cell/'manifest.json';q=cell/'predictions.csv.gz'
 if not p.exists() or not q.exists():return None
 m=json.loads(p.read_text())
 if m.get('status')!='complete' or m.get('run_type')!='formal':return None
 if m.get('data_sha256')!=digest_file(ROOT/f'data/processed/mimic_iv_{window}h.parquet'):return None
 if kind=='tabular' and (m.get('protocol_sha256')!=digest_file(run/'analysis_protocol.json') or m.get('feature_columns')!=GROUPS):return None
 if kind=='gru_seq' and m.get('sequence_sha256')!=digest_file(ROOT/'data/processed/mimic_iv_hourly.npz'):return None
 d=pd.read_csv(q,usecols=['pair_id','y_true','pred_D','pred_DT','pred_F']);y=d.y_true.to_numpy(dtype=int)
 if d.pair_id.duplicated().any() or len(np.unique(y))!=2:return None
 prediction_hash=digest_file(q)
 cache=run/'interval_cache'/kind/task/model/f'seed{seed}'/f'{window}h.json'
 if cache.exists():
  saved=json.loads(cache.read_text())
  if saved.get('prediction_sha256')==prediction_hash and saved.get('n_bootstrap')==n_boot:return saved['intervals']
 pred={g:d[f'pred_{g}'].to_numpy() for g in ['D','DT','F']}
 pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
 rng=np.random.default_rng(1000+seed+window*100+TASKS.index(task)*1000+(MODELS.index(model) if kind=='tabular' else 5)*10000)
 samples={'odi_raw':np.empty(n_boot),'odi_residual':np.empty(n_boot),'reference_gain':np.empty(n_boot)}
 for b in range(n_boot):
  idx=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
  yy=y[idx];auc={g:auc_fast(yy,pred[g][idx]) for g in pred}
  samples['odi_raw'][b]=auc['F']-auc['D'];samples['odi_residual'][b]=auc['F']-auc['DT'];samples['reference_gain'][b]=auc['DT']-auc['D']
 out={'task':task,'model':model,'window_h':window,'seed':seed,'n_test':len(d),'n_positive':len(pos),'run_id':run.name,'n_bootstrap':n_boot,
      'resampling_unit':'patient','resampling_scheme':'stratified_within_outcome','interval_method':'percentile_2.5_97.5'}
 for name,x in samples.items():
  lo,hi=np.quantile(x,[.025,.975]);out[f'{name}_ci_low']=float(lo);out[f'{name}_ci_high']=float(hi)
 cache.parent.mkdir(parents=True,exist_ok=True)
 tmp=cache.with_suffix('.tmp')
 tmp.write_text(json.dumps({'prediction_sha256':prediction_hash,'n_bootstrap':n_boot,'intervals':out},ensure_ascii=False)+'\n')
 os.replace(tmp,cache)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--kind',choices=['tabular','gru_seq'],default='tabular');p.add_argument('--workers',type=int,default=6);p.add_argument('--n-bootstrap',type=int,default=500);p.add_argument('--max-cells',type=int);a=p.parse_args();run=a.run_dir.resolve()
 if a.kind=='tabular':cells=list(itertools.product(TASKS,MODELS,WINDOWS,SEEDS))
 else:cells=[(t,'gru_seq',w,s) for t,w,s in itertools.product(TASKS,WINDOWS,SEEDS)]
 if a.max_cells:cells=cells[:a.max_cells]
 jobs=[(run,a.kind,*cell,a.n_bootstrap) for cell in cells];rows=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,f in enumerate(as_completed([pool.submit(one,x) for x in jobs]),1):
   r=f.result()
   if r:rows.append(r)
   if i%25==0 or i==len(jobs):print(i,'/',len(jobs),'interval rows',len(rows),flush=True)
 name='main_paired_intervals.csv.gz' if a.kind=='tabular' else 'gru_seq_paired_intervals.csv.gz'
 pd.DataFrame(rows).sort_values(['task','model','window_h','seed']).to_csv(run/name,index=False,compression='gzip') if rows else pd.DataFrame().to_csv(run/name,index=False,compression='gzip')
 print('wrote',name,len(rows),flush=True)
if __name__=='__main__':main()
