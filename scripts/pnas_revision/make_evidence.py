"""制作不含原始患者键的小型数据与实验证据包，并解压复算。"""
from __future__ import annotations
import argparse,hashlib,hmac,json,os,tempfile,zipfile,sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
CORE=['analysis_protocol.json','revision_status.json','input_inventory.csv','cohort_counts.csv','split_counts.csv',
      'age_counts.csv','hospital_counts.csv','field_time_rules.csv','sequence_feasibility.json',
      'data_audit_report.md','treatment_repair_manifest.json','historical_grid_comparison.csv',
      'main_seed_metrics.csv.gz','main_configuration_metrics.csv','missing_grid.csv',
      'age_seed_metrics.csv.gz','external_seed_metrics.csv.gz','subgroup_missing_grid.csv','subgroup_status.json',
      'gru_seq_seed_metrics.csv.gz','gru_seq_missing_grid.csv','manuscript_numbers.json','data_dictionary.json',
      'manuscript_before_after.md','temperature_repair_manifest.json','feature_schema.csv.gz','rerun_plan.csv',
      'rerun_summary.json','figure_data_map.csv','figure_captions_en.md','figure_revision_report_zh.md',
      'main_paired_intervals.csv.gz','gru_seq_paired_intervals.csv.gz','gru_seq_age_seed_metrics.csv.gz',
      'gru_seq_external_seed_metrics.csv.gz','gru_seq_subgroup_missing_grid.csv']

def sha_bytes(data):return hashlib.sha256(data).hexdigest()

def clinical_prediction(p,salt):
 d=pd.read_csv(p); required=['pair_id','y_true','pred_D','pred_O','pred_DT','pred_F']
 if any(x not in d for x in required):raise ValueError('正式预测字段缺失')
 d=d[required].copy()
 d['pair_id']=[hmac.new(salt,str(x).encode(),hashlib.sha256).hexdigest()[:20] for x in d.pair_id]
 return d

def old_prediction(salt):
 root=ROOT/'results/exp1/icu_mortality';joined=None
 for group,col in [('disease','pred_D'),('observation','pred_O'),('mixed','pred_F')]:
  p=root/group/'lr/seed42/24h_predictions.csv';d=pd.read_csv(p)
  d=d[['patient_id','y_true','y_pred']].rename(columns={'y_pred':col})
  if joined is None:joined=d
  else:joined=joined.merge(d,on=['patient_id','y_true'],validate='one_to_one')
 joined=joined.rename(columns={'patient_id':'pair_id'})
 joined['pair_id']=[hmac.new(salt,str(x).encode(),hashlib.sha256).hexdigest()[:20] for x in joined.pair_id]
 return joined[['pair_id','y_true','pred_D','pred_O','pred_F']]

def csv_bytes(df):return df.to_csv(index=False,float_format='%.17g').encode()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
 salt=os.urandom(32);payload={};omitted=[];examples=[]
 for name in CORE:
  path=run/name
  if path.exists():payload[f'core/{name}']=path.read_bytes()
  else:omitted.append({'path':name,'reason':'not yet produced'})
 for task,model,window,seed in [('icu_mortality','lr',24,42),('icu_mortality','lr',3,42),('icu_mortality','xgb',12,42)]:
  cell=run/'formal/tabular'/task/model/f'seed{seed}'/f'{window}h'
  man=cell/'manifest.json';pred=cell/'predictions.csv.gz'
  if not man.exists() or not pred.exists() or json.loads(man.read_text()).get('status')!='complete':
   omitted.append({'path':str(cell.relative_to(run)),'reason':'formal cell incomplete'});continue
  d=clinical_prediction(pred,salt)
  name=f'predictions/{task}_{model}_{window}h_seed{seed}.csv'
  payload[name]=csv_bytes(d)
  examples.append({'path':name,'complete_test_set':True,'n':len(d),'positive':int(d.y_true.sum()),'fields':list(d),'source_manifest':str(man.relative_to(run))})
  payload[f'cell_manifests/{task}_{model}_{window}h_seed{seed}.json']=man.read_bytes()
 old=old_prediction(salt);payload['predictions/historical_100_sample_lr_24h.csv']=csv_bytes(old)
 examples.append({'path':'predictions/historical_100_sample_lr_24h.csv','complete_test_set':False,'n':len(old),
                  'positive':int(old.y_true.sum()),'fields':list(old),'note':'完整历史 smoke 子集；非完整正式测试集；缺少 D+T 预测'})
 for task in ['icu_mortality']:
  cell=run/'formal/gru_seq'/task/'seed42'/'3h';man=cell/'manifest.json';pred=cell/'predictions.csv.gz'
  if man.exists() and pred.exists() and json.loads(man.read_text()).get('status')=='complete':
   d=clinical_prediction(pred,salt);name='predictions/icu_mortality_gru_seq_3h_seed42.csv';payload[name]=csv_bytes(d)
   examples.append({'path':name,'complete_test_set':True,'n':len(d),'positive':int(d.y_true.sum()),'fields':list(d)})
  else:omitted.append({'path':str(cell.relative_to(run)),'reason':'GRU-seq cell incomplete'})
 recompute='''from pathlib import Path\nimport pandas as pd\nfrom sklearn.metrics import roc_auc_score\nroot=Path(__file__).resolve().parent\nfor p in sorted((root/'predictions').glob('*.csv')):\n d=pd.read_csv(p);y=d.y_true\n print(p.name,'n=',len(d),'positive=',int(y.sum()))\n for group in ['D','O','DT','F']:\n  if 'pred_'+group in d:print(' ',group,roc_auc_score(y,d['pred_'+group]))\n'''
 payload['recompute.py']=recompute.encode()
 for script in ['audit.py','aggregate.py','aggregate_subgroups.py','aggregate_gru_subgroups.py','bootstrap_intervals.py','feature_schema.py','rerun_plan.py','make_evidence.py']:
  path=ROOT/'scripts/pnas_revision'/script
  payload[f'scripts/{script}']=path.read_bytes()
 readme=f'''# PNAS 数据与实验证据包\n\n运行标识：{run.name}。数据来源为本地 MIMIC-IV 2.2、eICU-CRD 2.0，处理队列已在 `data/processed/` 覆盖式修复，原始数据不入包。\n\n`core/` 保存输入清单、队列和划分审计、时间规则、协议、异常对照及已完成的正式指标；`predictions/` 是少量完整测试集或明确标注的历史 100 人子集。`cell_manifests/` 记录对应模型来源，`scripts/` 保存主要查询、导出和复算脚本。所有预测的患者键为包内一致的 HMAC 替代编号，密钥与原始映射均不入包。\n\n正式主分析只有五种表格模型；GRU-seq 单独统计。`missing_grid.csv` 和 `subgroup_status.json` 说明未完成项。科学定义待决项见审查报告，不可将跨库差异解释为因果效应。\n\n在包根目录运行 `python recompute.py` 可从完整预测复算 AUROC。历史 100 人文件只能复算该子集的 D、O、F 指标，不能视为完整正式测试集。\n'''
 payload['README_中文.md']=readme.encode()
 meta={'run_id':run.name,'predictions':examples,'omitted':omitted,'files':{name:{'bytes':len(data),'sha256':sha_bytes(data)} for name,data in payload.items()}}
 payload['manifest.json']=json.dumps(meta,ensure_ascii=False,indent=2).encode()
 output=run/'PNAS_Data_Evidence.zip';tmp=output.with_suffix('.tmp.zip')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
  for name,data in payload.items():z.writestr(name,data)
 if tmp.stat().st_size>30*2**20:tmp.unlink();raise ValueError('证据包超过 30 MiB 上限')
 os.replace(tmp,output)
 with zipfile.ZipFile(output) as z:
  for name,info in meta['files'].items():
   if sha_bytes(z.read(name))!=info['sha256']:raise ValueError(f'压缩包校验失败 {name}')
  for e in examples:
   d=pd.read_csv(z.open(e['path']))
   if len(d)!=e['n'] or int(d.y_true.sum())!=e['positive']:raise ValueError('预测复算失败')
 print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'mib':output.stat().st_size/2**20,'sha256':sha_bytes(output.read_bytes()),'prediction_files':len(examples),'omitted':omitted},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
