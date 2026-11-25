"""逐格复用已拟合模型，计算年龄及外部医院结果，失败保留日志。"""
from __future__ import annotations
import argparse,itertools,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.pnas_revision.run_tabular import TASKS,MODELS,WINDOWS,SEEDS,GROUPS,digest_file

def one(job):
    run,task,model,window,seed=job
    cell=run/'formal/tabular'/task/model/f'seed{seed}'/f'{window}h'
    if not (cell/'manifest.json').exists():return {'job':list(job[1:]),'status':'missing_formal'}
    manifest=json.loads((cell/'manifest.json').read_text())
    if manifest.get('status')!='complete':return {'job':list(job[1:]),'status':'incomplete_formal'}
    if (manifest.get('protocol_sha256')!=digest_file(run/'analysis_protocol.json')
        or manifest.get('data_sha256')!=digest_file(ROOT/f'data/processed/mimic_iv_{window}h.parquet')
        or manifest.get('feature_columns')!=GROUPS):
        return {'job':list(job[1:]),'status':'stale_formal'}
    if (cell/'postprocess_manifest.json').exists():
        post=json.loads((cell/'postprocess_manifest.json').read_text())
        if post.get('status')=='complete' and post.get('formal_predictions_sha256')==manifest.get('predictions_sha256') and (cell/'external_predictions.csv.gz').exists():return {'job':list(job[1:]),'status':'cached'}
    log=run/'logs/postprocess'/f'{task}_{model}_{window}h_seed{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ROOT/'scripts/pnas_revision/postprocess_tabular.py'),'--run-dir',str(run),'--task',task,'--model',model,'--window',str(window),'--seed',str(seed)]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('w') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    return {'job':list(job[1:]),'status':'complete' if code==0 else 'failed','log':str(log.relative_to(ROOT))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);p.add_argument('--workers',type=int,default=2);a=p.parse_args();run=a.run_dir.resolve()
    jobs=[(run,*x) for x in itertools.product(TASKS,a.models,WINDOWS,SEEDS)]
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(one,j) for j in jobs]),1):
            r=f.result();results.append(r);print(i,len(jobs),r['job'],r['status'],flush=True)
    (run/f'postprocess_dispatch_{"_".join(a.models)}.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
