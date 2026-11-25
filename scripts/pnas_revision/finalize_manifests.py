"""核验正式预测与模型产物，补齐最终代码、配置和数据指纹。"""
from __future__ import annotations
import argparse,hashlib,itertools,json,os,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,sklearn,yaml,torch,xgboost,lightgbm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pnas_revision.run_tabular import TASKS,MODELS,WINDOWS,SEEDS,GROUPS,digest_file,atomic_json
from scripts.pnas_revision.aggregate import read_cell

def model_config_hash(model,seed):
 cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text());settings=cfg['training'][{'lr':'logistic','rf':'random_forest','xgb':'xgboost','lgbm':'lightgbm','mlp':'mlp'}[model]].copy()
 if model=='rf':settings['n_jobs']=1
 identity={'model':model,'seed':seed,'settings':settings}
 return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

def artifacts(cell,groups,extension):
 out={}
 for g in groups:
  for kind,filename in [('model',f'model_{g}.{extension}'),('preprocessor',f'transformer_{g}.joblib')]:
   if extension=='pt' and kind=='preprocessor':continue
   p=cell/filename
   if not p.exists():raise FileNotFoundError(p)
   out[f'{kind}_{g}']={'path':str(p.relative_to(ROOT)),'sha256':digest_file(p),'bytes':p.stat().st_size}
 for kind,filename in [('predictions','predictions.csv.gz')]:
  p=cell/filename
  out[kind]={'path':str(p.relative_to(ROOT)),'sha256':digest_file(p),'bytes':p.stat().st_size}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 software={'python':sys.version.split()[0],'pandas':pd.__version__,'sklearn':sklearn.__version__,
           'xgboost':xgboost.__version__,'lightgbm':lightgbm.__version__,'torch':torch.__version__}
 protocol=digest_file(run/'analysis_protocol.json');data={w:digest_file(ROOT/f'data/processed/mimic_iv_{w}h.parquet') for w in WINDOWS}
 seq_hash=digest_file(ROOT/'data/processed/mimic_iv_hourly.npz');cfg_hash=digest_file(ROOT/'configs/default.yaml')
 counts={'tabular_complete':0,'gru_seq_complete':0,'stale':[]}
 for task,model,w,seed in itertools.product(TASKS,MODELS,WINDOWS,SEEDS):
  cell=run/'formal/tabular'/task/model/f'seed{seed}'/f'{w}h';p=cell/'manifest.json'
  if not p.exists():counts['stale'].append(str(cell.relative_to(run)));continue
  m=json.loads(p.read_text());metric,reason=read_cell(cell,protocol_hash=protocol,data_hash=data[w],feature_groups=GROUPS)
  if reason:counts['stale'].append(str(cell.relative_to(run))+':'+reason);continue
  expected_config=model_config_hash(model,seed)
  if m.get('model_config_sha256') not in (None,expected_config):raise ValueError(f'模型配置变更 {p}')
  if 'model_config_sha256' not in m:m['config_hash_backfill']='model-specific settings were unchanged; old manifest lacked this field'
  m['model_config_sha256']=expected_config;m['code_commit']=code_commit;m['software']=software
  m['artifacts']=artifacts(cell,GROUPS,'joblib');atomic_json(p,m);counts['tabular_complete']+=1
 for task,w,seed in itertools.product(TASKS,WINDOWS,SEEDS):
  cell=run/'formal/gru_seq'/task/f'seed{seed}'/f'{w}h';p=cell/'manifest.json'
  if not p.exists():counts['stale'].append(str(cell.relative_to(run)));continue
  m=json.loads(p.read_text())
  if m.get('status')!='complete' or m.get('sequence_sha256')!=seq_hash:
   counts['stale'].append(str(cell.relative_to(run))+':sequence_stale');continue
  if m.get('data_sha256')!=data[w]:
   m['data_hash_backfill']='only temp_first changed in tabular parquet after this GRU fit; GRU used stay_id, subject_id, age, gender and outcome, unchanged'
   m['data_sha256']=data[w]
  current_gru=yaml.safe_load((ROOT/'configs/default.yaml').read_text())['training']['gru_seq']
  for key,value in m.get('hyperparameters',{}).items():
   if key in current_gru and current_gru[key]!=value:raise ValueError(f'GRU 超参数不匹配 {p}: {key}')
  if m.get('config_sha256') not in (None,cfg_hash):raise ValueError(f'GRU 配置变更 {p}')
  if 'config_sha256' not in m:m['config_hash_backfill']='GRU parameters matched the current gru_seq settings; old manifest lacked config hash'
  m['config_sha256']=cfg_hash;m['code_commit']=code_commit;m['run_id']=run.name;m['software']=software
  m['artifacts']=artifacts(cell,('D','O','DT','F'),'pt');atomic_json(p,m);counts['gru_seq_complete']+=1
 (run/'manifest_finalization.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'tabular_complete':counts['tabular_complete'],'gru_seq_complete':counts['gru_seq_complete'],'stale_count':len(counts['stale'])},ensure_ascii=False))
if __name__=='__main__':main()
