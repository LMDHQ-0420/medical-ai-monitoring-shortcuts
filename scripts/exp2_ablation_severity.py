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


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['prepare','match','evaluate','diagnosis','all'],default='all')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    p.add_argument('--bootstrap',type=int,default=200)
    a=p.parse_args()
    root=a.output_root.resolve()
    if a.stage in ('prepare','all'):
        build_clinical_state();build_diagnosis()
    if a.stage in ('match','all'):match_test(root)
    if a.stage in ('evaluate','all'):evaluate_test(root,a.bootstrap)
    if a.stage in ('diagnosis','all'):evaluate_diagnosis(root)


if __name__=='__main__':main()
