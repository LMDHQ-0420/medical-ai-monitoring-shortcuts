"""Experiment 3: Variable grouping sensitivity ablation.

Tests robustness by reassigning MIXED_FEATURES to D-Model (variant A)
and O-Model (variant B), then comparing ODI under each grouping.

Iterates over all tasks x model_types x time_windows (aligned with exp1's grid
scale, though not all combinations may have data yet).

Usage:
    python scripts/exp3_ablation_grouping.py --config configs/default.yaml
    python scripts/exp3_ablation_grouping.py --task icu_mortality --model xgb --window 24
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd

from src.exp3.grouping import VARIANTS, run_variant
from src.data.transforms import split_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default=None, help="Run single task only")
    parser.add_argument("--model", default=None, choices=["lr", "xgb", "gru", "rf", "lgbm", "mlp"],
                        help="Run single model only")
    parser.add_argument("--window", type=int, default=None, help="Run single window only")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config training.seed; also selects the seedN/ output subdir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    seed_dir = f"seed{cfg['training']['seed']}"

    processed_dir = Path("data/processed")

    tasks = [args.task] if args.task else cfg["tasks"]
    model_types = [args.model] if args.model else ["lr", "rf", "xgb", "lgbm", "gru", "mlp"]
    windows = [args.window] if args.window else cfg["time_windows"]

    for window_h in windows:
        mimic_path = processed_dir / f"mimic_iv_{window_h}h.parquet"
        if not mimic_path.exists():
            print(f"[SKIP] {mimic_path} not found.")
            continue
        df = pd.read_parquet(mimic_path)

        seed = cfg["training"]["seed"]
        ratios = tuple(cfg["training"]["train_val_test_split"])

        for model_type in model_types:
            for task in tasks:
                if task not in df.columns:
                    print(f"[SKIP] Label column '{task}' not found in parquet.")
                    continue

                out_path = (Path("results") / "exp3" / task / "mixed" / model_type / seed_dir /
                            f"{window_h}h_grouping_ablation.json")
                if out_path.exists():
                    print(f"[SKIP] {out_path} already exists.")
                    continue

                print(f"\n>>> task={task} | model={model_type} | window={window_h}h")
                task_df = df[df[task].notna()].copy()
                train_df, val_df, test_df = split_cohort(task_df, ratios=ratios, seed=seed)

                task_results = {}
                for variant_name, d_cols, o_cols in VARIANTS:
                    try:
                        odi_result = run_variant(train_df, val_df, test_df, task, model_type,
                                                  cfg, d_cols, o_cols)
                        task_results[variant_name] = odi_result
                        print(f"  {variant_name}: ODI={odi_result['odi']:.4f} "
                              f"[{odi_result['ci_lower']:.4f}, {odi_result['ci_upper']:.4f}]")
                    except Exception as e:
                        print(f"  {variant_name} failed: {e}")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(task_results, f, indent=2)
                print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
