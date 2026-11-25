"""只从已完成且可配对的正式预测重算五模型与序列扩展统计。"""
from __future__ import annotations
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pnas_revision.run_tabular import TASKS,MODELS,WINDOWS,SEEDS,GROUPS as FEATURE_COLUMNS,digest_file
PRED_GROUPS=('D','O','DT','F')


def read_cell(cell,protocol_hash=None,data_hash=None,feature_groups=None,sequence_hash=None):
    p=cell/'manifest.json';q=cell/'predictions.csv.gz'
    if not p.exists() or not q.exists():return None,'missing_manifest_or_prediction'
    m=json.loads(p.read_text())
    if m.get('status')!='complete' or m.get('run_type')!='formal':return None,'not_complete_formal'
    if protocol_hash is not None and m.get('protocol_sha256')!=protocol_hash:return None,'stale_protocol'
    if data_hash is not None and m.get('data_sha256')!=data_hash:return None,'stale_data'
    if feature_groups is not None and m.get('feature_columns')!=feature_groups:return None,'stale_feature_columns'
    if sequence_hash is not None and m.get('sequence_sha256')!=sequence_hash:return None,'stale_sequence'
    pred=pd.read_csv(q)
    if not {'pair_id','stay_id','y_true',*(f'pred_{x}' for x in PRED_GROUPS)}.issubset(pred):return None,'missing_paired_column'
    if pred.pair_id.duplicated().any() or pred.stay_id.duplicated().any():return None,'duplicate_pair_key'
    if len(pred)!=m.get('split_counts',{}).get('test',{}).get('n',m.get('n_test')):return None,'sample_count_mismatch'
    y=pred.y_true.to_numpy()
    if int(y.sum())!=m.get('split_counts',{}).get('test',{}).get('positive',m.get('positive_test')):return None,'event_count_mismatch'
    if len(np.unique(y))!=2:return None,'one_class_test'
    auc={g:float(roc_auc_score(y,pred[f'pred_{g}'])) for g in PRED_GROUPS}
    for g in PRED_GROUPS:
        if abs(auc[g]-m[f'auc_{g}'])>1e-8:return None,f'auc_{g}_mismatch'
    metric={'n_test':len(pred),'n_positive':int(y.sum()),'n_negative':int(len(pred)-y.sum()),
            **{f'auc_{g}':auc[g] for g in PRED_GROUPS},'odi_raw':auc['F']-auc['D'],
            'odi_residual':auc['F']-auc['DT'],'reference_gain':auc['DT']-auc['D'],
            'split_id':m['split_id'],'run_id':m.get('run_id',cell.parents[4].name),
            'prediction_file':str(q.relative_to(ROOT))}
    return metric,None


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
    protocol_hash=digest_file(run/'analysis_protocol.json')
    data_hashes={w:digest_file(ROOT/f'data/processed/mimic_iv_{w}h.parquet') for w in WINDOWS}
    sequence_path=ROOT/'data/processed/mimic_iv_hourly.npz'
    sequence_hash=digest_file(sequence_path) if sequence_path.exists() else None
    rows=[];missing=[];allman=[]
    for task,model,window,seed in itertools.product(TASKS,MODELS,WINDOWS,SEEDS):
        key={'task':task,'model':model,'window_h':window,'seed':seed}
        cell=run/'formal/tabular'/task/model/f'seed{seed}'/f'{window}h'
        metric,reason=read_cell(cell,protocol_hash=protocol_hash,data_hash=data_hashes[window],feature_groups=FEATURE_COLUMNS)
        if metric:rows.append({**key,**metric});allman.append({**key,'status':'complete','manifest':str((cell/'manifest.json').relative_to(ROOT))})
        else:missing.append({**key,'reason':reason});allman.append({**key,'status':reason})
    frame=pd.DataFrame(rows)
    frame.to_csv(run/'main_seed_metrics.csv.gz',index=False,compression='gzip',float_format='%.17g')
    pd.DataFrame(missing).to_csv(run/'missing_grid.csv',index=False)
    if len(frame):
        summary=frame.groupby(['task','model','window_h'],as_index=False).agg(
            seeds=('seed','nunique'),n_test_mean=('n_test','mean'),n_positive_mean=('n_positive','mean'),
            auc_D=('auc_D','mean'),auc_O=('auc_O','mean'),auc_DT=('auc_DT','mean'),auc_F=('auc_F','mean'),
            odi_raw=('odi_raw','mean'),odi_residual=('odi_residual','mean'),reference_gain=('reference_gain','mean'))
        summary.to_csv(run/'main_configuration_metrics.csv',index=False,float_format='%.17g')
    else:summary=pd.DataFrame()
    seq=[];seq_missing=[]
    for task,window,seed in itertools.product(TASKS,WINDOWS,SEEDS):
        cell=run/'formal/gru_seq'/task/f'seed{seed}'/f'{window}h';metric,reason=read_cell(cell,data_hash=data_hashes[window],sequence_hash=sequence_hash)
        key={'task':task,'model':'gru_seq','window_h':window,'seed':seed}
        if metric:seq.append({**key,**metric})
        else:seq_missing.append({**key,'reason':reason})
    pd.DataFrame(seq).to_csv(run/'gru_seq_seed_metrics.csv.gz',index=False,compression='gzip')
    pd.DataFrame(seq_missing).to_csv(run/'gru_seq_missing_grid.csv',index=False)
    numbers={'run_id':run.name,'tabular_seed_cells_complete':len(frame),'tabular_seed_cells_expected':625,
             'tabular_configuration_cells_complete_5_seeds':int((summary.seeds==5).sum()) if len(summary) else 0,
             'tabular_configuration_cells_expected':125,'gru_seq_seed_cells_complete':len(seq),'gru_seq_seed_cells_expected':125,
             'main_odi_raw_positive_seed_cells':int((frame.odi_raw>0).sum()) if len(frame) else 0,
             'main_odi_raw_positive_configurations_5_seeds':int(((summary.seeds==5)&(summary.odi_raw>0)).sum()) if len(summary) else 0,
             'scientific_definition_status':'unresolved; descriptive outputs only'}
    (run/'manuscript_numbers.json').write_text(json.dumps(numbers,ensure_ascii=False,indent=2)+'\n')
    (run/'experiments_manifest.json').write_text(json.dumps(allman,ensure_ascii=False,indent=2)+'\n')
    dictionary={'unit':'seed-level row is one task × model × window × seed; configuration summary averages five seeds',
      'patient_count':'n_test is individuals in the test split, not independent configurations',
      'difference_formulas':{'odi_raw':'auc_F − auc_D','odi_residual':'auc_F − auc_DT','reference_gain':'auc_DT − auc_D'},
      'group_definitions':{'D':'14 measured vital/lab values plus age, gender, admission source','O':'30 explicit observation-process fields','T':'three treatment indicators','DT':'D+T','F':'D+O+T'},
      'excluded_late_availability_fields':['sofa_score','comorbidity flags','diagnosis category flags'],
      'interval_algorithm':'not computed until patient-level paired bootstrap is run; no interval columns in this table',
      'missing_reason_file':'missing_grid.csv','smoke_policy':'excluded by explicit formal path and manifest run_type'}
    (run/'data_dictionary.json').write_text(json.dumps(dictionary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(numbers,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
