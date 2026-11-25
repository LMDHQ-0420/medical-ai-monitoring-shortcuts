"""五模型正式格点调度，独立日志，失败继续，重复调用自动跳过有效缓存。"""
from __future__ import annotations
import argparse,itertools,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TASKS=('icu_mortality','inhosp_mortality','aki','sepsis_deterioration','escalation')
WINDOWS=(3,6,12,18,24);SEEDS=(42,43,44,45,46);MODELS=('lr','rf','xgb','lgbm','mlp')

def run_one(arg):
    run_dir,task,model,window,seed=arg
    log_dir=run_dir/'logs'/'tabular';log_dir.mkdir(parents=True,exist_ok=True)
    log=log_dir/f'{task}_{model}_{window}h_seed{seed}.log'
    cmd=[sys.executable,str(ROOT/'scripts/pnas_revision/run_tabular.py'),'--run-dir',str(run_dir),'--task',task,'--model',model,'--window',str(window),'--seed',str(seed)]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    with log.open('w') as f:
        p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
    return {'task':task,'model':model,'window_h':window,'seed':seed,'exit_code':p.returncode,'log':str(log.relative_to(ROOT)),'command':' '.join(cmd)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);p.add_argument('--workers',type=int,default=4);p.add_argument('--max-cells',type=int);a=p.parse_args()
    run_dir=a.run_dir.resolve()
    cells=list(itertools.product(TASKS,a.models,WINDOWS,SEEDS))
    if a.max_cells:cells=cells[:a.max_cells]
    args=[(run_dir,*cell) for cell in cells]
    outputs=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_one,cell):cell for cell in args}
        for i,fut in enumerate(as_completed(futures),1):
            result=fut.result();outputs.append(result)
            print(f"{i}/{len(cells)} {result['task']} {result['model']} {result['window_h']}h seed{result['seed']} exit={result['exit_code']}",flush=True)
    path=run_dir/f'dispatch_tabular_{"_".join(a.models)}.json';path.write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
    print('dispatch complete; failed:',sum(x['exit_code']!=0 for x in outputs),flush=True)
if __name__=='__main__':main()
