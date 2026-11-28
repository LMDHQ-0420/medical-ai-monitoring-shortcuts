"""实验 2 正式入口：调用统一流程的已冻结评价阶段。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.run_formal import evaluate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/processed/landmark_v2_staging')
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/.landmark_v2_staging')
    args=parser.parse_args()
    protocol=json.loads((args.output_root/'analysis_protocol.json').read_text())
    evaluate(args.output_root,args.data_root,protocol)

if __name__=='__main__':main()
