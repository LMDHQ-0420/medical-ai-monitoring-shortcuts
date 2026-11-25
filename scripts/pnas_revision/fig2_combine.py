"""合并可续跑的五任务匹配评估分片并核对格点覆盖。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TASKS = ("icu_mortality", "inhosp_mortality", "aki", "sepsis_deterioration", "escalation")


def combine(run: Path) -> None:
    for stem, expected in (("matched_seed_metrics", 625), ("context_seed_metrics", None)):
        parts = []
        for task in TASKS:
            path = run / f"{stem}_{task}.csv.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            part = pd.read_csv(path)
            if stem == "matched_seed_metrics" and len(part) != 125:
                raise ValueError(f"{task} 的匹配格点只有 {len(part)}/125")
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        if expected is not None and len(frame) != expected:
            raise ValueError(f"{stem} 合并后行数错误")
        key = ["task", "model", "window_h", "seed"]
        if stem == "context_seed_metrics":
            key += ["stratum"]
        if frame.duplicated(key).any():
            raise ValueError(f"{stem} 格点重复")
        frame.to_csv(run / f"{stem}.csv.gz", index=False, compression="gzip")
        print(stem, len(frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    combine(parser.parse_args().run_dir.resolve())
