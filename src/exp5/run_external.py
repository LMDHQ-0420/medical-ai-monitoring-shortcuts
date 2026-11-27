"""实验 4/5 共用正式模型完成年龄与外部评估；结束后只保留汇总表。"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import pandas as pd
from src.exp1.trainer import MODELS,TASKS,WINDOWS,SEEDS
from src.exp5.external_validation import run as evaluate_tabular
from src.utils.logger import experiment_log

DERIVED=('age_metrics.csv','external_predictions.csv.gz',
         'hospital_metrics.csv','postprocess_manifest.json')


def collect(root:Path,kind:str)->pd.DataFrame:
    rows=[]
    cells=((t,m,w,s,root/'models/tabular'/t/m/f'seed{s}'/f'{w}h')
           for t,m,w,s in itertools.product(TASKS,MODELS,WINDOWS,SEEDS))
    name='age_metrics.csv' if kind=='exp4' else 'hospital_metrics.csv'
    for task,model,window,seed,cell in cells:
        path=cell/name
        if not path.exists():raise FileNotFoundError(path)
        rows.extend({'task':task,'model':model,'window_h':window,'seed':seed,**row}
                    for row in pd.read_csv(path).to_dict('records'))
    return pd.DataFrame(rows)


def run_experiment(root:Path,task:str|None,model:str|None,window:int|None,seed:int|None):
    tasks=(task,) if task else TASKS
    models=(model,) if model else MODELS
    windows=(window,) if window else WINDOWS
    seeds=(seed,) if seed is not None else SEEDS
    for t,m,w,s in itertools.product(tasks,models,windows,seeds):
        with experiment_log(('exp4','exp5'),f'{t}_{m}_seed{s}_{w}h',detail=f'output_root={root}; shared_age_external_evaluation=True'):
            evaluate_tabular(root,t,m,w,s)
    # 指定单格时只保存该格中间结果，避免用局部结果覆盖完整正式表。
    if any(value is not None for value in (task,model,window,seed)):
        print('单格已完成；完整五模型网格完成后再汇总 exp4/exp5')
        return
    with experiment_log(('exp4','exp5'),'summary',detail=f'output_root={root}'):
        for kind in ('exp4','exp5'):
            out=root/kind;out.mkdir(parents=True,exist_ok=True)
            collect(root,kind).to_csv(out/'results.csv.gz',index=False,compression='gzip')
        for path in (root/'models').rglob('*'):
            if path.is_file() and path.name in DERIVED:path.unlink()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=TASKS)
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--window',type=int,choices=WINDOWS)
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[2]/'outputs')
    a=p.parse_args()
    run_experiment(a.output_root.resolve(),a.task,a.model,a.window,a.seed)
