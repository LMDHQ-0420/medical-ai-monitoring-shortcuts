"""冻结权重外部推理入口；评估与汇总由独立 collect 阶段执行。"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.exp1.trainer import TASKS,MODELS,WINDOWS,SEEDS
from src.data.feature_groups import PREDICTOR_SETS
from src.exp5.external_validation import infer_group


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--task',choices=TASKS)
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--window',type=int,choices=WINDOWS)
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--predictor-set',choices=tuple(PREDICTOR_SETS))
    a=p.parse_args()
    protocol=json.loads(a.protocol.read_text())
    grid=itertools.product([a.task] if a.task else TASKS,
                           [a.model] if a.model else MODELS,
                           [a.window] if a.window else WINDOWS,
                           [a.seed] if a.seed is not None else SEEDS,
                           [a.predictor_set] if a.predictor_set else PREDICTOR_SETS)
    for task,model,window,seed,predictor_set in grid:
        result=infer_group(a.data_root,a.output_root,protocol,task,model,window,seed,predictor_set)
        print(task,model,window,seed,predictor_set,result['source'],flush=True)

if __name__=='__main__':main()
