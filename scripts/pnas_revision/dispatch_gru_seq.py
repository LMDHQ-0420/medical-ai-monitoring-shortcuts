"""按 GPU 分配真实序列 GRU 格点，逐格日志与缓存续跑。"""
from __future__ import annotations
import argparse,itertools,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TASKS=('icu_mortality','inhosp_mortality','aki','sepsis_deterioration','escalation')
WINDOWS=(3,6,12,18,24);SEEDS=(42,43,44,45,46)

def one(job):
 run,task,window,seed,gpu=job
 log=run/'logs/gru_seq'/f'{task}_{window}h_seed{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(ROOT/'scripts/pnas_revision/run_gru_seq.py'),'--run-dir',str(run),'--task',task,'--window',str(window),'--seed',str(seed)]
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with log.open('w') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 return {'task':task,'window_h':window,'seed':seed,'gpu':gpu,'exit_code':code,'log':str(log.relative_to(ROOT))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3]);p.add_argument('--workers',type=int,default=4);a=p.parse_args();run=a.run_dir.resolve()
 jobs=[(run,*cell,a.gpus[i%len(a.gpus)]) for i,cell in enumerate(itertools.product(TASKS,WINDOWS,SEEDS))]
 done=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,f in enumerate(as_completed([pool.submit(one,x) for x in jobs]),1):
   r=f.result();done.append(r);print(i,'/',len(jobs),r,flush=True)
 (run/'dispatch_gru_seq.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n')
 print('GRU-seq failed:',sum(x['exit_code']!=0 for x in done),flush=True)
if __name__=='__main__':main()
