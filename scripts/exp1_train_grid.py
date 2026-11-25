"""Train EHR models for experiment 1: D-Model vs O-Model vs M-Model.

Iterates over all tasks x feature_groups x model_types x time_windows.

Usage:
    python scripts/exp1_train_grid.py --config configs/default.yaml [--smoke-test]
    python scripts/exp1_train_grid.py --task icu_mortality --model xgb --window 24
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd

from src.exp1.trainer import EHRTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default=None, help="Run single task only")
    parser.add_argument("--model", default=None, choices=["lr", "xgb", "rf", "lgbm", "mlp"],
                        help="Run single model only")
    parser.add_argument("--window", type=int, default=None, help="Run single window only")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config training.seed; also selects the seedN/ output subdir")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Single task/model/window with 500 samples")
    parser.add_argument("--output-root", default="outputs/legacy_exp1",
                        help="新运行的结果根目录；历史 results/ 保持只读")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    seed_dir = f"seed{cfg['training']['seed']}"

    processed_dir = Path("data/processed")
    output_root = Path(args.output_root) / ("smoke" if args.smoke_test else "formal")

    tasks = [args.task] if args.task else cfg["tasks"]
    model_types = [args.model] if args.model else ["lr", "rf", "xgb", "lgbm", "mlp"]
    windows = [args.window] if args.window else cfg["time_windows"]
    feature_groups = ["disease", "observation", "mixed"]

    if args.smoke_test:
        tasks = [tasks[0]]
        model_types = ["lr"]
        windows = [windows[-1]]

    for window_h in windows:
        mimic_path = processed_dir / f"mimic_iv_{window_h}h.parquet"
        if not mimic_path.exists():
            print(f"[SKIP] {mimic_path} not found.")
            continue
        df = pd.read_parquet(mimic_path)
        if args.smoke_test:
            df = df.head(500)

        for task in tasks:
            if task not in df.columns:
                print(f"[SKIP] Label column '{task}' not found in parquet.")
                continue
            for fg in feature_groups:
                for mt in model_types:
                    out_path = output_root / task / fg / mt / seed_dir / f"{window_h}h_metrics.json"
                    if out_path.exists():
                        print(f"[SKIP] {out_path} already exists.")
                        continue
                    print(f"\n>>> task={task} | group={fg} | model={mt} | window={window_h}h")
                    trainer = EHRTrainer(
                        model_type=mt,
                        task=task,
                        feature_group=fg,
                        window_h=window_h,
                        config=cfg,
                        output_base=str(output_root),
                        seed_dir=seed_dir,
                    )
                    metrics = trainer.run(df)
                    print(f"    AUROC={metrics['auroc']:.4f} "
                          f"[{metrics['auroc_ci_lower']:.4f}, {metrics['auroc_ci_upper']:.4f}] "
                          f"| Brier={metrics['brier']:.4f}")


if __name__ == "__main__":
    main()
