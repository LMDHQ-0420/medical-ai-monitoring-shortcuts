"""正式实验唯一总入口：prepare→plan→train→infer→evaluate→collect→validate。"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import itertools
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.data.feature_groups import PREDICTOR_SETS
from src.exp1.trainer import MODELS,TASKS,WINDOWS,SEEDS,fit_group
from src.formal.splits import make_registry, audit_legacy_conflicts
from src.formal.identity import file_hash

WORKER={}


def init_worker(config_path, data_root, output_root, protocol_path, registry_path):
    WORKER.update(config=yaml.safe_load(Path(config_path).read_text()),
                  data_root=Path(data_root), output_root=Path(output_root),
                  protocol=json.loads(Path(protocol_path).read_text()),
                  registry=pd.read_parquet(registry_path))


def fit_job(args):
    task,model,window_h,seed,state,predictor_set=args
    result=fit_group(WORKER['config'],WORKER['data_root'],WORKER['output_root'],WORKER['protocol'],
                     task,model,window_h,seed,state,predictor_set,WORKER['registry'])
    return (task,model,window_h,seed,state,predictor_set,result['fit_source'])


def infer_job(args):
    from src.exp5.external_validation import infer_group
    task,model,window_h,seed,predictor_set=args
    result=infer_group(WORKER['data_root'],WORKER['output_root'],WORKER['protocol'],
                       task,model,window_h,seed,predictor_set)
    return (task,model,window_h,seed,predictor_set,result['source'])


def jobs():
    for task,model,window,seed,predictor_set in itertools.product(TASKS,MODELS,WINDOWS,SEEDS,PREDICTOR_SETS):
        yield task,model,window,seed,'first_measurements',predictor_set
    for model,seed,predictor_set in itertools.product(('lr','lgbm'),SEEDS,('clinical_state_treatment','full_model')):
        yield 'inhosp_mortality',model,24,seed,'first_plus_latest',predictor_set


def prepare(data_root:Path, output_root:Path):
    from src.exp2.clinical_state import build as build_state
    from src.exp2.diagnosis_timing import build as build_diagnosis
    from src.formal.splits import split_frame
    from src.formal.identity import canonical_hash
    fig_manifest=output_root/'fig_checksum_before.txt'
    if not fig_manifest.exists():
        from src.formal.identity import file_hash
        lines=[f'{file_hash(path)}  {path.relative_to(ROOT)}'
               for path in sorted((ROOT/'fig').rglob('*')) if path.is_file()]
        fig_manifest.parent.mkdir(parents=True,exist_ok=True)
        fig_manifest.write_text('\n'.join(lines)+'\n')
    registry=make_registry(ROOT/'data/processed/mimic_iv_3h.parquet',
                           data_root/'mimic_iv_3h.parquet',data_root/'split_registry.parquet')
    audit_legacy_conflicts(ROOT/'data/processed',registry,TASKS,WINDOWS,output_root/'legacy_split_conflicts.csv')
    state = build_state(data_root)
    if state is None:
        state = pd.read_parquet(data_root / 'mimic_iv_clinical_state.parquet')
    coverage=[]
    for window in WINDOWS:
        subset=state.loc[state.window_h.eq(window)]
        cohort=pd.read_parquet(data_root / f'mimic_iv_{window}h.parquet',columns=['stay_id'])
        if len(subset)!=len(cohort) or set(subset.stay_id)!=set(cohort.stay_id):
            raise ValueError(f'{window}h 最新状态侧表与队列不一致')
        for vital in ('hr','sbp','dbp','temp','rr','spo2'):
            observed=subset[f'{vital}_observed'].fillna(False).astype(bool)
            available=pd.to_datetime(subset.loc[observed,f'{vital}_available_time'])
            chart=pd.to_datetime(subset.loc[observed,f'{vital}_charttime'])
            cutoff=pd.to_datetime(subset.loc[observed,'prediction_cutoff'])
            if (available>=cutoff).any() or (chart>available).any():
                raise ValueError(f'{window}h {vital} 最新状态违反截点可用性')
            coverage.append({'window_h':window,'vital':vital,'n':len(subset),
                             'n_observed':int(observed.sum()),'coverage':float(observed.mean())})
    (output_root/'exp2').mkdir(parents=True,exist_ok=True)
    pd.DataFrame(coverage).to_csv(output_root/'exp2/state_control_coverage.csv',index=False)
    build_diagnosis(data_root)
    impact=[]
    for dataset in ('mimic_iv','eicu'):
        for window in WINDOWS:
            old=pd.read_parquet(ROOT/f'data/processed/{dataset}_{window}h.parquet')
            new=pd.read_parquet(data_root/f'{dataset}_{window}h.parquet')
            old_ids=set(old.stay_id)
            new_ids=set(new.stay_id)
            for task in TASKS:
                known=new[task].notna()
                impact.append({'dataset':dataset,'window_h':window,'task':task,
                               'n_old':len(old),'n_new':len(new),'new_stays':len(new_ids-old_ids),
                               'removed_stays':len(old_ids-new_ids),
                               'old_positive':int(old[task].sum()),
                               'new_known':int(known.sum()),'new_unknown':int((~known).sum()),
                               'new_positive':int(new.loc[known,task].sum())})
    report=output_root/'data_impact.csv'
    report.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(impact).to_csv(report,index=False)
    source={'protocol_id':'landmark-v2-20261128','processed':{},'raw':{}}
    for path in sorted(data_root.glob('*.parquet')):
        source['processed'][path.name]={'sha256':file_hash(path),'bytes':path.stat().st_size}
    for dataset, path in {'mimic_iv':ROOT/'data/raw/mimic_iv/mimic-iv-2.2/SHA256SUMS.txt',
                          'eicu':ROOT/'data/raw/eicu/eicu-collaborative-research-database-2.0/SHA256SUMS.txt'}.items():
        source['raw'][dataset]={'checksum_manifest':str(path.relative_to(ROOT)),
                                'checksum_manifest_sha256':file_hash(path),
                                'individual_archive_sha256_verified':False}
    (output_root/'data_source_manifest.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
    print('prepared',len(registry),'split rows',len(impact),'impact rows',flush=True)


def plan(config_path,data_root,output_root,protocol_path,registry_path):
    init_worker(config_path,data_root,output_root,protocol_path,registry_path)
    counts={}
    for args in jobs():
        result=fit_group(WORKER['config'],WORKER['data_root'],WORKER['output_root'],
                         WORKER['protocol'],*args,WORKER['registry'],action='plan')
        key=(result['fit_action'],result['prediction_action'])
        counts[key]=counts.get(key,0)+1
    print('plan',counts,flush=True)
    return counts


def parallel_stage(action, config_path,data_root,output_root,protocol_path,registry_path,workers):
    initargs=(str(config_path),str(data_root),str(output_root),str(protocol_path),str(registry_path))
    items=list(jobs()) if action=='train' else [(*j[:4],j[5]) for j in jobs() if j[4]=='first_measurements']
    function=fit_job if action=='train' else infer_job
    done=0
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=initargs) as pool:
        future={pool.submit(function,item):item for item in items}
        for completed in as_completed(future):
            item=future[completed]
            try:
                result=completed.result()
            except Exception as exc:
                print('FAILED',item,repr(exc),flush=True)
                raise
            done+=1
            if done%10==0 or done==len(items):
                print(action,'complete',done,'remaining',len(items)-done,'latest',result,flush=True)


def evaluate(output_root:Path,data_root:Path,protocol:dict):
    from src.exp1.collect import collect as exp1
    from src.exp2.matching import run as match
    from src.exp2.evaluate import collect as exp2
    from src.exp2.diagnosis_strata import collect as diagnosis
    from src.exp4.age_transfer import collect as exp4
    from src.exp5.collect import collect as exp5
    registry=pd.read_parquet(data_root/'split_registry.parquet')
    print('exp1',exp1(output_root,protocol),flush=True)
    match(output_root,data_root,registry)
    print('exp2 matching',exp2(output_root,data_root,protocol),flush=True)
    print('exp5',exp5(output_root,protocol),flush=True)
    print('exp2 diagnosis',diagnosis(output_root,data_root,protocol),flush=True)
    print('exp4',exp4(output_root,protocol),flush=True)
    from src.exp2.state_control import collect as state
    print('exp2 state',state(output_root,protocol),flush=True)


def collect(output_root:Path,protocol:dict):
    from src.exp3.grouping import collect as exp3
    print('exp3',exp3(output_root,protocol),flush=True)


def validate(data_root:Path,output_root:Path,protocol:dict):
    from src.exp1.trainer import group_dir
    failures=[]
    count=0
    registry_rows=[]
    for task,model,window,seed,state,predictor_set in jobs():
        cell=group_dir(output_root,task,model,window,seed,state,predictor_set)
        try:
            manifest=json.loads((cell/'manifest.json').read_text())
            for name,key in [('model.joblib','model_sha256'),('transformer.joblib','transformer_sha256'),
                             ('internal_predictions.csv.gz','prediction_sha256')]:
                if file_hash(cell/name)!=manifest[key]:
                    raise ValueError(name+' SHA mismatch')
            if manifest['status']!='complete' or manifest['analysis_id']!=protocol['analysis_id']:
                raise ValueError('manifest incomplete')
            count+=1
            registry_rows.append({key:manifest.get(key) for key in ('task','model','window_h','seed','state_representation','predictor_set','fit_id','prediction_id','evaluation_id','cohort_id','split_id','endpoint_version','analysis_id','fit_source','n_train','n_validation','n_test','n_positive_test','model_sha256','transformer_sha256','prediction_sha256')})
        except Exception as exc:
            failures.append({'cell':str(cell),'error':repr(exc)})
    required={'exp1':['model_metrics.csv.gz','gain_metrics.csv.gz'],
              'exp2':['balance.csv','matched_metrics.csv.gz','matched_contrasts.csv.gz','state_control_metrics.csv.gz','state_control_contrasts.csv.gz','state_control_coverage.csv','context_metrics.csv.gz','diagnosis_metrics.csv.gz'],
              'exp3':['gain_components.csv.gz','transfer_components.csv.gz','transfer_summary.csv.gz'],
              'exp4':['age_metrics.csv.gz','age_contrasts.csv.gz','age_transfer_contrasts.csv.gz','age_eligibility.csv'],
              'exp5':['hospital_metrics.csv.gz','hospital_reversal.csv.gz','hospital_eligibility.csv','probability_metrics.csv.gz','probability_contrasts.csv.gz']}
    external_count=0
    for task,model,window,seed,state,predictor_set in jobs():
        if state!='first_measurements':
            continue
        cell=group_dir(output_root,task,model,window,seed,state,predictor_set)
        try:
            manifest=json.loads((cell/'external_manifest.json').read_text())
            if manifest['status']!='complete' or file_hash(cell/'external_predictions.csv.gz')!=manifest['prediction_sha256']:
                raise ValueError('外部概率来源不完整')
            external_count+=1
        except Exception as exc:
            failures.append({'cell':str(cell),'error':'external: '+repr(exc)})
    for group,names in required.items():
        for name in names:
            path=output_root/group/name
            if not path.is_file():
                failures.append({'file':f'{group}/{name}','error':'missing'})
            elif path.stat().st_size==0:
                failures.append({'file':f'{group}/{name}','error':'empty'})
    if (data_root/'split_registry.parquet').is_file():
        registry=pd.read_parquet(data_root/'split_registry.parquet')
        if registry.duplicated(['subject_id','seed']).any() or registry.split.isna().any():
            failures.append({'file':'split_registry.parquet','error':'患者划分重复或缺失'})
    else:
        failures.append({'file':'split_registry.parquet','error':'missing'})
    for path,expected in ((output_root/'exp1/model_metrics.csv.gz',2500),
                          (output_root/'exp1/gain_metrics.csv.gz',625),
                          (output_root/'exp2/state_control_metrics.csv.gz',40),
                          (output_root/'exp2/state_control_contrasts.csv.gz',10)):
        if path.is_file() and len(pd.read_csv(path))!=expected:
            failures.append({'file':str(path),'error':f'行数不等于 {expected}'})
    fig_manifest=output_root/'fig_checksum_before.txt'
    if fig_manifest.is_file():
        for line in fig_manifest.read_text().splitlines():
            sha,relative=line.split('  ',1)
            path=ROOT/relative
            if not path.is_file() or file_hash(path)!=sha:
                failures.append({'file':relative,'error':'fig SHA changed'})
    else:
        failures.append({'file':'fig_checksum_before.txt','error':'missing'})
    from src.exp1.trainer import atomic_csv
    atomic_csv(output_root/'model_registry.csv.gz',pd.DataFrame(registry_rows))
    report={'protocol_id':protocol['protocol_id'],'models_complete':count,'models_expected':2520,
            'external_complete':external_count,'external_expected':2500,
            'failures':failures,'status':'complete' if count==2520 and external_count==2500 and not failures else 'failed'}
    (output_root/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('validation',report['status'],'models',count,'failures',len(failures),flush=True)
    if failures or count!=2520 or external_count!=2500:raise ValueError('正式结果验证失败')


def promote(data_root: Path, output_root: Path):
    """完整核验后以同盘重命名替换正式产物；异常时回滚已移动的旧版本。"""
    report = json.loads((output_root / 'validation_report.json').read_text())
    if report.get('status') != 'complete':
        raise ValueError('仅允许已核验的 staging 结果正式替换')
    backup = output_root / 'promotion_backup'
    if backup.exists():
        raise ValueError(f'存在未处理的替换备份: {backup}')
    moves = []
    for source in sorted(data_root.iterdir()):
        if source.is_file() and source.suffix in ('.parquet', '.json'):
            moves.append((source, ROOT / 'data/processed' / source.name))
    for relative in ('analysis_protocol.json', 'data_impact.csv', 'data_source_manifest.json',
                     'validation_report.json', 'models/tabular', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5'):
        source = output_root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        moves.append((source, ROOT / 'outputs' / relative))
    completed = []
    try:
        for source, destination in moves:
            old = backup / destination.relative_to(ROOT)
            old.parent.mkdir(parents=True, exist_ok=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            had_old = destination.exists()
            if had_old:
                os.replace(destination, old)
            try:
                os.replace(source, destination)
            except BaseException:
                if had_old:
                    os.replace(old, destination)
                raise
            completed.append((source, destination, old, had_old))
        model_root = ROOT / 'outputs/models/tabular'
        from src.exp1.trainer import atomic_json
        for manifest_path in model_root.rglob('manifest.json'):
            manifest = json.loads(manifest_path.read_text())
            manifest['data_root'] = str(ROOT / 'data/processed')
            manifest['output_root'] = str(ROOT / 'outputs')
            manifest['model_artifact_root'] = str(ROOT / 'outputs')
            atomic_json(manifest_path, manifest)
        release = {'protocol_id': report['protocol_id'], 'status': 'complete',
                   'models_complete': report['models_complete'],
                   'external_complete': report['external_complete'],
                   'source_validation_report': 'outputs/validation_report.json'}
        (ROOT / 'outputs/formal_release_manifest.json').write_text(json.dumps(release, ensure_ascii=False, indent=2) + '\n')
    except BaseException:
        for source, destination, old, had_old in reversed(completed):
            os.replace(destination, source)
            if had_old:
                os.replace(old, destination)
        raise
    shutil.rmtree(backup)
    print('promoted', len(completed), 'formal artifacts', flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('prepare','plan','train','infer','evaluate','collect','validate','promote','run'),required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/default.yaml')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/processed/landmark_v2_staging')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/.landmark_v2_staging')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    data_root=a.data_root.resolve();output_root=a.output_root.resolve()
    protocol_path=output_root/'analysis_protocol.json'
    if not protocol_path.exists():
        output_root.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'configs/analysis_protocol.json',protocol_path)
    protocol=json.loads(protocol_path.read_text())
    registry=data_root/'split_registry.parquet'
    for stage in (('prepare','plan','train','infer','evaluate','collect','validate','promote') if a.stage=='run' else (a.stage,)):
        if stage=='prepare':prepare(data_root,output_root)
        elif stage=='plan':plan(a.config,data_root,output_root,protocol_path,registry)
        elif stage in ('train','infer'):parallel_stage(stage,a.config,data_root,output_root,protocol_path,registry,a.workers)
        elif stage=='evaluate':evaluate(output_root,data_root,protocol)
        elif stage=='collect':collect(output_root,protocol)
        elif stage=='validate':validate(data_root,output_root,protocol)
        elif stage=='promote':promote(data_root,output_root)

if __name__=='__main__':main()
