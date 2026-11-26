"""治疗重分组复用实验 1 的同一完整模型预测。"""
from __future__ import annotations
import shutil
from pathlib import Path


def reuse_formal_results(output_root:Path)->Path:
    source=output_root/'exp1/results.csv.gz'
    if not source.is_file():raise FileNotFoundError(source)
    target=output_root/'exp3/results.csv.gz'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    return target
