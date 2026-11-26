"""复用正式模型与配对预测，计算年龄分层及逐医院外部评估。"""
from __future__ import annotations
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
from src.exp1.trainer import GROUPS,digest_file
from src.exp4.age_transfer import age_label,contrasts

def predict_external(cell:Path,eicu:pd.DataFrame,task:str)->pd.DataFrame:
    """从同一正式权重生成 eICU 患者级预测，不重新拟合。"""
    ext=pd.DataFrame({'pair_id':eicu.subject_id.to_numpy(),'stay_id':eicu.stay_id.to_numpy(),
                      'hospital_id':eicu.hospital_id.to_numpy(),'age':eicu.age.to_numpy(),
                      'y_true':eicu[task].astype(int).to_numpy()})
    for g,cols in GROUPS.items():
        transformer=joblib.load(cell/f'transformer_{g}.joblib')
        fitted=joblib.load(cell/f'model_{g}.joblib')
        ext[f'pred_{g}']=np.asarray(fitted.predict_proba(transformer.transform(eicu))).reshape(-1)
    return ext


def run(run_dir,task,model,window,seed):
    cell=run_dir/'models/tabular'/task/model/f'seed{seed}'/f'{window}h'
    manifest=json.loads((cell/'manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('run_type')!='formal':raise ValueError('只接受已完成正式格点')
    if (manifest.get('protocol_sha256')!=digest_file(run_dir/'analysis_protocol.json') or manifest.get('data_sha256')!=digest_file(ROOT/f'data/processed/mimic_iv_{window}h.parquet') or manifest.get('feature_columns')!=GROUPS):
        raise ValueError('正式格点与当前协议、数据或特征组不匹配')
    internal=pd.read_csv(cell/'predictions.csv.gz')
    mimic=pd.read_parquet(ROOT/'data/processed'/f'mimic_iv_{window}h.parquet',columns=['subject_id','age'])
    mimic=mimic.rename(columns={'subject_id':'pair_id'})
    internal=internal.merge(mimic,on='pair_id',validate='one_to_one')
    internal['age_group']=age_label(internal.age)
    age_rows=[]
    for level,sub in internal.groupby('age_group',observed=True):
        c=contrasts(sub)
        if c:age_rows.append({'dataset':'mimic_iv','age_group':str(level),**c})
    eicu=pd.read_parquet(ROOT/'data/processed'/f'eicu_{window}h.parquet')
    eicu=eicu.loc[eicu[task].notna()].copy()
    ext=predict_external(cell,eicu,task)
    ext['age_group']=age_label(ext.age)
    out=cell/'external_predictions.csv.gz';tmp=cell/'external_predictions.tmp.csv.gz';ext.to_csv(tmp,index=False,compression='gzip',float_format='%.17g');tmp.replace(out)
    hospital_rows=[]
    for hospital,sub in ext.groupby('hospital_id'):
        c=contrasts(sub)
        if c:hospital_rows.append({'hospital_id':hospital,**c,
                                   'internal_odi_raw':manifest['odi_raw'],
                                   'odi_shrinkage':manifest['odi_raw']-c['odi_raw']})
    pd.DataFrame(hospital_rows).to_csv(cell/'hospital_metrics.csv',index=False)
    for level,sub in ext.groupby('age_group',observed=True):
        c=contrasts(sub)
        if c:age_rows.append({'dataset':'eicu','age_group':str(level),**c})
    pd.DataFrame(age_rows).to_csv(cell/'age_metrics.csv',index=False)
    info={'status':'complete','formal_predictions_sha256':manifest['predictions_sha256'],'external_label_compatibility':'unresolved for AKI, Deterioration and Escalation; risk set unresolved for mortality',
          'valid_hospitals':len(hospital_rows),'external_rows':len(ext),'age_group_bounds':'<45,45-64,65-80,>80'}
    (cell/'postprocess_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(task,model,window,seed,'hospitals',len(hospital_rows),flush=True)
