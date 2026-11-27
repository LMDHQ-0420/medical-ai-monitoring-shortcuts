"""实验 2：截点前临床状态匹配及诊断录入分层。复用 exp1 正式模型。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.exp2.clinical_state import build as build_clinical_state
from src.exp2.diagnosis_timing import build as build_diagnosis
from src.exp2.matching import run as match_test
from src.exp2.evaluate import run as evaluate_test
from src.exp2.diagnosis_strata import run as evaluate_diagnosis
from src.utils.logger import experiment_log


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['prepare','match','evaluate','diagnosis','all'],default='all')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    p.add_argument('--bootstrap',type=int,default=200)
    a=p.parse_args()
    root=a.output_root.resolve()
    stages={
        'prepare':lambda:(build_clinical_state(),build_diagnosis()),
        'match':lambda:match_test(root),
        'evaluate':lambda:evaluate_test(root,a.bootstrap),
        'diagnosis':lambda:evaluate_diagnosis(root),
    }
    for stage,run in stages.items():
        if a.stage in (stage,'all'):
            with experiment_log('exp2',stage,detail=f'output_root={root}'):
                run()


if __name__=='__main__':main()
