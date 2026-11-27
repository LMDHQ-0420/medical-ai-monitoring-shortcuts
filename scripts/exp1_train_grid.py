"""统一正式训练入口：五种表格模型共用患者划分与输出目录。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.exp1.trainer import MODELS,TASKS,WINDOWS,SEEDS,run_cell
from src.utils.logger import experiment_log


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--window',type=int,choices=WINDOWS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    tasks=[args.task] if args.task else cfg['tasks']
    models=[args.model] if args.model else list(MODELS)
    windows=[args.window] if args.window else cfg['time_windows']
    seeds=[args.seed] if args.seed is not None else SEEDS
    if args.smoke_test:
        tasks,models,windows,seeds=tasks[:1],['lr'],windows[-1:],seeds[:1]
    root=args.output_root.resolve()
    for task in tasks:
        for model in models:
            for window in windows:
                for seed in seeds:
                    name=f'{task}_{model}_seed{seed}_{window}h'
                    if args.smoke_test:name='smoke_'+name
                    with experiment_log('exp1',name,detail=f'output_root={root}'):
                        run_cell(root,task,model,window,seed,smoke=args.smoke_test)


if __name__=='__main__':main()
