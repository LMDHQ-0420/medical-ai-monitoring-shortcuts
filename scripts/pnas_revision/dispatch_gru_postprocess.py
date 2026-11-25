"""逐格复用已拟合 GRU 权重完成真实序列年龄与外部医院评估。"""
from __future__ import annotations
import argparse,itertools,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TASKS=('icu_mortality','inhosp_mortality','aki','sepsis_deterioration','escalation')
WINDOWS=(3,6,12,18,24);SEEDS=(42,43,44,45,46)

def one(job):
 run,task,window,seed=job;cell=run/'formal/gru_seq'/task/f'seed{seed}'/f'{window}h'
 m=cell/'manifest.json'
 if not m.exists() or json.loads(m.read_text()).get('status')!='complete':return {'task':task,'window_h':window,'seed':seed,'status':'training_incomplete'}
 log=run/'logs/gru_postprocess'/f'{task}_{window}h_seed{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(ROOT/'scripts/pnas_revision/postprocess_gru_seq.py'),'--run-dir',str(run),'--task',task,'--window',str(window),'--seed',str(seed)]
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with log.open('w') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 return {'task':task,'window_h':window,'seed':seed,'status':'complete' if code==0 else 'failed','log':str(log.relative_to(ROOT))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args();run=a.run_dir.resolve()
 jobs=[(run,*x) for x in itertools.product(TASKS,WINDOWS,SEEDS)];results=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,f in enumerate(as_completed([pool.submit(one,j) for j in jobs]),1):
   r=f.result();results.append(r);print(i,'/',len(jobs),r,flush=True)
 (run/'dispatch_gru_postprocess.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
