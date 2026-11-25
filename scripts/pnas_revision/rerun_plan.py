"""按当前数据、划分协议和特征组生成最小必要续跑清单。"""
from __future__ import annotations
import argparse,itertools,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pnas_revision.run_tabular import TASKS,MODELS,WINDOWS,SEEDS,GROUPS,digest_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve();protocol=digest_file(run/'analysis_protocol.json')
 data={w:digest_file(ROOT/f'data/processed/mimic_iv_{w}h.parquet') for w in WINDOWS};rows=[]
 for task,model,w,seed in itertools.product(TASKS,MODELS,WINDOWS,SEEDS):
  cell=run/'formal/tabular'/task/model/f'seed{seed}'/f'{w}h';p=cell/'manifest.json';pred=cell/'predictions.csv.gz'
  reason='missing';status='rerun_required';reuse=''
  if p.exists():
   m=json.loads(p.read_text());reuse=','.join(m.get('reused_groups',[]))
   if m.get('status')!='complete':reason='failed_or_interrupted'
   elif m.get('run_type')!='formal':reason='smoke_identity'
   elif m.get('data_sha256')!=data[w]:reason='data_fingerprint_mismatch'
   elif m.get('protocol_sha256')!=protocol:reason='protocol_fingerprint_mismatch'
   elif m.get('feature_columns')!=GROUPS:reason='feature_order_mismatch'
   elif not pred.exists() or m.get('predictions_sha256')!=digest_file(pred):reason='prediction_missing_or_hash_mismatch'
   else:reason='verified_complete';status='complete'
  rows.append({'task':task,'model':model,'window_h':w,'seed':seed,'status':status,'reason':reason,'verified_reused_groups':reuse,
               'manifest_path':str(p.relative_to(ROOT)) if p.exists() else ''})
 pd.DataFrame(rows).to_csv(run/'rerun_plan.csv',index=False)
 summary={'expected_seed_cells':625,'current_complete':sum(x['status']=='complete' for x in rows),
          'rerun_required':sum(x['status']!='complete' for x in rows),
          'historical_results_policy':'not reusable: old six-model schema, input-order split and unverified 100-sample smoke products'}
 (run/'rerun_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
