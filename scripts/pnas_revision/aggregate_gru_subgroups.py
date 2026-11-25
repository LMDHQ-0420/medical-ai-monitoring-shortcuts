"""独立汇总 GRU-seq 的年龄与逐医院外部结果，并标明未完成格点。"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import pandas as pd
TASKS=('icu_mortality','inhosp_mortality','aki','sepsis_deterioration','escalation')
WINDOWS=(3,6,12,18,24);SEEDS=(42,43,44,45,46)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
 age=[];ext=[];missing=[]
 for task,window,seed in itertools.product(TASKS,WINDOWS,SEEDS):
  key={'task':task,'model':'gru_seq','window_h':window,'seed':seed,'run_id':run.name}
  cell=run/'formal/gru_seq'/task/f'seed{seed}'/f'{window}h';p=cell/'postprocess_manifest.json';fit=cell/'manifest.json'
  if not p.exists() or not fit.exists():missing.append({**key,'reason':'missing_postprocess'});continue
  j=json.loads(p.read_text());m=json.loads(fit.read_text())
  if j.get('status')!='complete' or j.get('internal_predictions_sha256')!=m.get('predictions_sha256'):
   missing.append({**key,'reason':'stale_or_incomplete_postprocess'});continue
  age.extend([{**key,**r} for r in pd.read_csv(cell/'age_metrics.csv').to_dict('records')])
  ext.extend([{**key,**r} for r in pd.read_csv(cell/'external_hospital_metrics.csv').to_dict('records')])
 pd.DataFrame(age).to_csv(run/'gru_seq_age_seed_metrics.csv.gz',index=False,compression='gzip')
 pd.DataFrame(ext).to_csv(run/'gru_seq_external_seed_metrics.csv.gz',index=False,compression='gzip')
 pd.DataFrame(missing).to_csv(run/'gru_seq_subgroup_missing_grid.csv',index=False)
 print('GRU postprocess complete cells',125-len(missing),'age rows',len(age),'hospital rows',len(ext))
if __name__=='__main__':main()
