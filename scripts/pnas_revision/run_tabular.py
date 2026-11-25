"""五模型主分析：统一 D/O/D+T/F 预测，支持按格点续跑。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.feature_groups import DISEASE_FEATURES,OBSERVATION_FEATURES,MIXED_FEATURES
from src.data.transforms import EHRTransformer,split_cohort
from src.models.train_utils import create_and_fit

GROUPS={"D":DISEASE_FEATURES,"O":OBSERVATION_FEATURES,"DT":DISEASE_FEATURES+MIXED_FEATURES,"F":DISEASE_FEATURES+OBSERVATION_FEATURES+MIXED_FEATURES}
MODELS=("lr","rf","xgb","lgbm","mlp")
TASKS=("icu_mortality","inhosp_mortality","aki","sepsis_deterioration","escalation")
WINDOWS=(3,6,12,18,24)
SEEDS=(42,43,44,45,46)


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
    return h.hexdigest()


def digest_values(values):
    return hashlib.sha256('\n'.join(map(str,sorted(values))).encode()).hexdigest()


def atomic_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
    os.replace(temp,path)


def cell_path(run_dir,task,model,window,seed,smoke=False):
    return run_dir/('smoke' if smoke else 'formal')/'tabular'/task/model/f'seed{seed}'/f'{window}h'

def paired_cache_matches(saved,expected,manifest,task):
    labels=expected[task].astype(int).to_numpy()
    return (len(saved)==len(expected)
            and manifest.get('split_counts',{}).get('test',{}).get('n')==len(expected)
            and manifest.get('split_counts',{}).get('test',{}).get('positive')==int(labels.sum())
            and np.array_equal(saved.pair_id.to_numpy(),expected.subject_id.to_numpy())
            and np.array_equal(saved.stay_id.to_numpy(),expected.stay_id.to_numpy())
            and np.array_equal(saved.y_true.to_numpy(),labels))

def run_cell(run_dir,task,model,window,seed,smoke=False):
    if model not in MODELS or task not in TASKS or window not in WINDOWS or seed not in SEEDS:
        raise ValueError('格点不属于固定协议')
    data_path=ROOT/'data/processed'/f'mimic_iv_{window}h.parquet'
    if not data_path.exists(): raise FileNotFoundError(data_path)
    data_hash=digest_file(data_path)
    protocol=run_dir/'analysis_protocol.json'
    protocol_hash=digest_file(protocol)
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
    cfg['training']['seed']=seed
    # 并行格点各占一个线程；仅修改运行并行度，不改变树的统计参数。
    cfg['training']['random_forest']['n_jobs']=1
    config_identity={'model':model,'seed':seed,'settings':cfg['training'][{'lr':'logistic','rf':'random_forest','xgb':'xgboost','lgbm':'lightgbm','mlp':'mlp'}[model]]}
    model_config_sha256=hashlib.sha256(json.dumps(config_identity,sort_keys=True).encode()).hexdigest()
    cell_dir=cell_path(run_dir,task,model,window,seed,smoke)
    cell_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=cell_dir/'manifest.json'
    old=None
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        pred=cell_dir/'predictions.csv.gz'
        if (old.get('status')=='complete' and old.get('run_type')==('smoke' if smoke else 'formal')
            and old.get('data_sha256')==data_hash and old.get('protocol_sha256')==protocol_hash
            and old.get('model_config_sha256')==model_config_sha256
            and old.get('seed')==seed and old.get('task')==task and old.get('model')==model
            and old.get('window_h')==window and pred.exists() and digest_file(pred)==old.get('predictions_sha256')):
            # 缓存还必须对应当前队列的完整测试集，而非仅凭预测文件存在。
            check=pd.read_parquet(data_path,columns=['subject_id','stay_id',task]).loc[lambda x:x[task].notna()]
            _,_,expected=split_cohort(check,seed=seed)
            saved=pd.read_csv(pred,usecols=['pair_id','stay_id','y_true'])
            valid=paired_cache_matches(saved,expected,old,task)
            if valid:
                print('VALID CACHE',cell_dir,flush=True);return old
            print('REJECT CACHE: patient set, labels, or sample count differ',cell_dir,flush=True)
    df=pd.read_parquet(data_path)
    df=df.loc[df[task].notna()].copy()
    if smoke: df=df.head(500).copy()
    train,val,test=split_cohort(df,seed=seed)
    if len(set(train.subject_id)&set(test.subject_id)) or len(set(train.subject_id)&set(val.subject_id)) or len(set(val.subject_id)&set(test.subject_id)):
        raise ValueError('患者划分交叉')
    if test[task].nunique()<2: raise ValueError('测试集中只有单一结局类别')
    labels=test[task].astype(int).to_numpy()
    pred=pd.DataFrame({'pair_id':test.subject_id.to_numpy(),'stay_id':test.stay_id.to_numpy(),'y_true':labels})
    # 协议分组修正时仅复用患者、标签、数据和特征均未改变的 D/O 拟合。
    reusable={}
    old_pred_path=cell_dir/'predictions.csv.gz'
    if (old and old.get('status')=='complete' and old.get('data_sha256')==data_hash
        and old.get('run_type')==('smoke' if smoke else 'formal')
        and old.get('task')==task and old.get('model')==model and old.get('seed')==seed
        and old.get('window_h')==window and old.get('split_id')==digest_values(test.subject_id.unique())
        and old_pred_path.exists() and digest_file(old_pred_path)==old.get('predictions_sha256')):
        previous=pd.read_csv(old_pred_path)
        aligned=(len(previous)==len(test)
                 and np.array_equal(previous.pair_id.to_numpy(),test.subject_id.to_numpy())
                 and np.array_equal(previous.stay_id.to_numpy(),test.stay_id.to_numpy())
                 and np.array_equal(previous.y_true.to_numpy(),labels))
        if aligned:
            for group in ('D','O'):
                if (old.get('feature_columns',{}).get(group)==GROUPS[group]
                    and (cell_dir/f'model_{group}.joblib').exists()
                    and (cell_dir/f'transformer_{group}.joblib').exists()):
                    reusable[group]=previous[f'pred_{group}'].to_numpy()
    if reusable: print('REUSE VERIFIED GROUPS',list(reusable),cell_dir,flush=True)
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest={'status':'running','run_type':'smoke' if smoke else 'formal','task':task,'model':model,'window_h':window,'seed':seed,
              'run_id':run_dir.name,'input_commit':git_head,'data_path':str(data_path.relative_to(ROOT)),
              'data_sha256':data_hash,'protocol_sha256':protocol_hash,'model_config_sha256':model_config_sha256,
              'split_id':digest_values(test.subject_id.unique()),
              'split_counts':{name:{'n':len(part),'positive':int(part[task].sum()),'patients':part.subject_id.nunique()} for name,part in [('train',train),('validation',val),('test',test)]},
              'feature_columns':GROUPS,'software':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__},
              'failure_reason':None}
    atomic_json(manifest_path,manifest)
    try:
        for group,cols in GROUPS.items():
            if group in reusable:
                score=reusable[group]
                pred[f'pred_{group}']=score
                manifest[f'auc_{group}']=float(roc_auc_score(labels,score))
                manifest.setdefault('reused_groups',[]).append(group)
                continue
            transformer=EHRTransformer(cols)
            x_train=transformer.fit_transform(train)
            x_val=transformer.transform(val)
            x_test=transformer.transform(test)
            fitted=create_and_fit(model,cfg,x_train,train[task].to_numpy(),x_val,val[task].to_numpy())
            score=np.asarray(fitted.predict_proba(x_test)).reshape(-1)
            if len(score)!=len(test): raise ValueError('预测长度错误')
            pred[f'pred_{group}']=score
            joblib.dump(transformer,cell_dir/f'transformer_{group}.joblib')
            joblib.dump(fitted,cell_dir/f'model_{group}.joblib')
            manifest[f'auc_{group}']=float(roc_auc_score(labels,score))
            print(task,model,window,seed,group,manifest[f'auc_{group}'],flush=True)
        manifest['odi_raw']=manifest['auc_F']-manifest['auc_D']
        manifest['odi_residual']=manifest['auc_F']-manifest['auc_DT']
        manifest['reference_gain']=manifest['auc_DT']-manifest['auc_D']
        if abs(manifest['odi_raw']-manifest['odi_residual']-manifest['reference_gain'])>1e-12:
            raise ValueError('ODI 恒等式错误')
        temp=cell_dir/'predictions.tmp.csv.gz'
        pred.to_csv(temp,index=False,compression='gzip',float_format='%.17g')
        os.replace(temp,cell_dir/'predictions.csv.gz')
        manifest['predictions_sha256']=digest_file(cell_dir/'predictions.csv.gz')
        manifest['status']='complete'
    except Exception as exc:
        manifest['status']='failed'
        manifest['failure_reason']=repr(exc)
        manifest['traceback']=traceback.format_exc()
        atomic_json(manifest_path,manifest)
        raise
    atomic_json(manifest_path,manifest)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--window',type=int,choices=WINDOWS,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run_cell(a.run_dir.resolve(),a.task,a.model,a.window,a.seed,a.smoke)

if __name__=='__main__':main()
