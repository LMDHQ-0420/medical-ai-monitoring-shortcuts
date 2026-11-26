"""实验 3：使用实验 1 的同一批 D、D+T、F 预测，不重复拟合 F。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.exp3.grouping import reuse_formal_results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    a=p.parse_args()
    print(reuse_formal_results(a.output_root.resolve()))


if __name__=='__main__':main()
