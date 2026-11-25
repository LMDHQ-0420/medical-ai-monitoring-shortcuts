"""从覆盖后的实际队列生成字段分组、单位与每窗口缺失情况。"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.feature_groups import DISEASE_FEATURES,OBSERVATION_FEATURES,MIXED_FEATURES,LABEL_COLS
UNITS={'hr_first':'beats/min','sbp_first':'mmHg','dbp_first':'mmHg','temp_first':'Celsius for values mapped from 80–115 F; other outliers unresolved',
       'rr_first':'breaths/min','spo2_first':'percent','creatinine_first':'mg/dL (source dependent)',
       'lactate_first':'mmol/L (source dependent)','wbc_first':'10^3/uL','hgb_first':'g/dL','sodium_first':'mEq/L',
       'potassium_first':'mEq/L','bilirubin_first':'mg/dL','inr_first':'ratio','age':'years'}

EXCLUDED={'sofa_score','has_diabetes','has_ckd','has_heart_failure','has_copd','has_cancer','dx_infection','dx_cardiovascular','dx_respiratory','dx_renal','dx_other'}

def group(c):
 if c in EXCLUDED:return 'excluded_late_availability'
 if c in DISEASE_FEATURES:return 'D'
 if c in OBSERVATION_FEATURES:return 'O'
 if c in MIXED_FEATURES:return 'T'
 if c in LABEL_COLS:return 'outcome'
 if c=='blood_culture_sent_24h':return 'unused_duplicate_observation'
 return 'identifier_or_metadata'

def unit(c):
 if c in UNITS:return UNITS[c]
 if c.endswith('_first'):return 'source dependent'
 if c.startswith(('vitals_count','labs_count','orders_count','abg_count','cxr_count')) or 'repeat' in c:return 'count'
 if c in LABEL_COLS or c.startswith(('has_','dx_')) or c.startswith(('antibiotics_','vasopressors_','mechanical_','blood_culture_')):return 'binary'
 return 'not applicable or source dependent'

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();rows=[]
 for dataset in ['mimic_iv','eicu']:
  for window in [3,6,12,18,24]:
   path=ROOT/f'data/processed/{dataset}_{window}h.parquet';df=pd.read_parquet(path)
   for c in df:
    rows.append({'dataset':dataset,'window_h':window,'column':c,'group':group(c),'unit':unit(c),
                 'dtype':str(df[c].dtype),'n_rows':len(df),'n_missing':int(df[c].isna().sum()),
                 'missing_fraction':float(df[c].isna().mean()),'n_nonzero':int((df[c].fillna(0)!=0).sum()) if pd.api.types.is_numeric_dtype(df[c]) else None})
 pd.DataFrame(rows).to_csv(a.run_dir.resolve()/'feature_schema.csv.gz',index=False,compression='gzip')
 print('schema rows',len(rows))
if __name__=='__main__':main()
