"""正式主模型入口；配置和处理数据目录显式传入统一 trainer。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.exp1.trainer import MODELS, TASKS, WINDOWS, SEEDS, fit_group
from src.data.feature_groups import PREDICTOR_SETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--split-registry", type=Path, required=True)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--window", type=int, choices=WINDOWS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--predictor-set", choices=tuple(PREDICTOR_SETS))
    p.add_argument("--action", choices=("plan", "train", "infer"), default="train")
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    protocol = json.loads(args.protocol.read_text())
    registry = pd.read_parquet(args.split_registry)
    for task in ([args.task] if args.task else TASKS):
        for model in ([args.model] if args.model else MODELS):
            for window in ([args.window] if args.window else WINDOWS):
                for seed in ([args.seed] if args.seed is not None else SEEDS):
                    for predictor_set in ([args.predictor_set] if args.predictor_set else PREDICTOR_SETS):
                        result = fit_group(config, args.data_root, args.output_root, protocol,
                                           task, model, window, seed, "first_measurements",
                                           predictor_set, registry, action=args.action)
                        print(task, model, window, seed, predictor_set,
                              result.get("fit_action", result.get("fit_source")), flush=True)

if __name__ == "__main__":
    main()
