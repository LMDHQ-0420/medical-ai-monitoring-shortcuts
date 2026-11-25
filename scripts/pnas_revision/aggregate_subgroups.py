"""汇总真实年龄与逐医院外部预测；缺失的匹配/诊断实验仅输出空表及原因。"""
from __future__ import annotations
import argparse,itertools,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pnas_revision.run_tabular import TASKS,MODELS,WINDOWS,SEEDS


def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
 age=[];external=[];missing=[]
 for task,model,window,seed in itertools.product(TASKS,MODELS,WINDOWS,SEEDS):
  key={'task':task,'model':model,'window_h':window,'seed':seed,'run_id':run.name}
  cell=run/'formal/tabular'/task/model/f'seed{seed}'/f'{window}h'
  manifest=cell/'postprocess_manifest.json'
  formal=cell/'manifest.json'
  if not manifest.exists() or not formal.exists():
   missing.append({**key,'reason':'postprocess_incomplete'});continue
  post=json.loads(manifest.read_text());fit=json.loads(formal.read_text())
  if post.get('status')!='complete' or post.get('formal_predictions_sha256')!=fit.get('predictions_sha256'):
   missing.append({**key,'reason':'postprocess_stale_or_incomplete'});continue
  ap=cell/'age_metrics.csv';ep=cell/'hospital_metrics.csv'
  if not ap.exists() or not ep.exists():missing.append({**key,'reason':'metrics_file_missing'});continue
  age.extend([{**key,**r} for r in pd.read_csv(ap).to_dict('records')])
  external.extend([{**key,**r} for r in pd.read_csv(ep).to_dict('records')])
 pd.DataFrame(age).to_csv(run/'age_seed_metrics.csv.gz',index=False,compression='gzip')
 pd.DataFrame(external).to_csv(run/'external_seed_metrics.csv.gz',index=False,compression='gzip')
 pd.DataFrame(missing).to_csv(run/'subgroup_missing_grid.csv',index=False)
 pd.DataFrame(columns=['task','model','window_h','seed','matched_odi_raw','n_matched','reason']).to_csv(run/'matched_seed_metrics.csv.gz',index=False,compression='gzip')
 pd.DataFrame(columns=['task','model','window_h','seed','diagnosis','odi_raw','n','reason']).to_csv(run/'diagnosis_seed_metrics.csv.gz',index=False,compression='gzip')
 reasons={'matched':'not rerun: landmark-available severity and matching protocol require scientific definition',
          'diagnosis':'not rerun: diagnosis availability by prediction landmark is unresolved',
          'external':'exploratory until MIMIC/eICU labels and risk sets are harmonized'}
 (run/'subgroup_status.json').write_text(json.dumps({'complete_postprocess_cells':625-len(missing),'age_rows':len(age),'external_hospital_rows':len(external),'blockers':reasons},ensure_ascii=False,indent=2)+'\n')
 print('postprocessed cells',625-len(missing),'age rows',len(age),'external rows',len(external))
if __name__=='__main__':main()
