"""Experiment 5: Age-based ODI inequity (mimic_gap), and whether it survives
transfer to eICU.

Trains mixed + disease models on MIMIC (same procedure as exp4), then computes
ODI = AUROC(M) - AUROC(D) by age group on:
  (a) MIMIC-internal test set
  (b) pooled eICU (all hospitals together, since per-hospital age strata are
      too small for stable AUROC)

mimic_gap = ODI(>80) - ODI(<45) is the core exp5 statistic; the eICU-side gap
and its change on transfer are reported as exploratory, not part of the core
support-rate statistic.

Usage:
    python scripts/exp5_age_odi_transfer.py --config configs/default.yaml
    python scripts/exp5_age_odi_transfer.py --task icu_mortality --model xgb --window 24
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import pandas as pd

from src.exp5.age_transfer import odi_by_age, age_gap
from src.data.transforms import split_cohort, prepare_xy
from src.data.feature_groups import DISEASE_FEATURES, OBSERVATION_FEATURES, MIXED_FEATURES
from src.models.train_utils import create_and_fit

MIXED_FEATURE_COLS = DISEASE_FEATURES + OBSERVATION_FEATURES + MIXED_FEATURES


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
        eicu_path = processed_dir / f"eicu_{window_h}h.parquet"
        if not mimic_path.exists() or not eicu_path.exists():
            print(f"[SKIP] window={window_h}h: parquet not found.")
            continue

        mimic_df = pd.read_parquet(mimic_path)
        eicu_df = pd.read_parquet(eicu_path)

        seed = cfg["training"]["seed"]
        ratios = tuple(cfg["training"]["train_val_test_split"])

        for model_type in model_types:
            task_gaps = {}
            for task in tasks:
                if task not in mimic_df.columns or task not in eicu_df.columns:
                    print(f"[SKIP] Label column '{task}' not found in parquet.")
                    continue

                out_path = (Path("results") / "exp5" / task / "mixed" / model_type / seed_dir /
                            f"{window_h}h_odi_by_age_transfer.csv")
                if out_path.exists():
                    print(f"[SKIP] {out_path} already exists.")
                    continue

                print(f"\n>>> task={task} | model={model_type} | window={window_h}h")
                task_df = mimic_df[mimic_df[task].notna()].copy()
                train_df, val_df, test_df = split_cohort(task_df, ratios=ratios, seed=seed)

                X_tr_m, y_tr_m, tr_m = prepare_xy(train_df, MIXED_FEATURE_COLS, task, fit=True)
                X_val_m, y_val_m, _ = prepare_xy(val_df, MIXED_FEATURE_COLS, task, transformer=tr_m)
                model_m = create_and_fit(model_type, cfg, X_tr_m, y_tr_m, X_val_m, y_val_m)

                X_tr_d, y_tr_d, tr_d = prepare_xy(train_df, DISEASE_FEATURES, task, fit=True)
                X_val_d, y_val_d, _ = prepare_xy(val_df, DISEASE_FEATURES, task, transformer=tr_d)
                model_d = create_and_fit(model_type, cfg, X_tr_d, y_tr_d, X_val_d, y_val_d)

                mimic_odi = odi_by_age(test_df, task, model_m, model_d, tr_m, tr_d)
                mimic_odi["source"] = "mimic_internal"
                eicu_odi = odi_by_age(eicu_df, task, model_m, model_d, tr_m, tr_d)
                eicu_odi["source"] = "eicu_external"

                combined = pd.concat([mimic_odi, eicu_odi], ignore_index=True)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(out_path, index=False)
                print(combined.to_string(index=False))

                mimic_gap = age_gap(mimic_odi)
                eicu_gap = age_gap(eicu_odi)
                task_gaps[task] = {
                    "mimic_gap": mimic_gap,
                    "eicu_gap": eicu_gap,
                    "gap_change_on_transfer": (
                        eicu_gap - mimic_gap
                        if not (np.isnan(mimic_gap) or np.isnan(eicu_gap)) else float("nan")
                    ),
                }
                print(f"  age-ODI gap (>80 minus <45): MIMIC={mimic_gap:.4f}, eICU={eicu_gap:.4f}")

            if not task_gaps:
                continue

            mimic_gaps = np.array([v["mimic_gap"] for v in task_gaps.values()])
            eicu_gaps = np.array([v["eicu_gap"] for v in task_gaps.values()])
            valid = ~(np.isnan(mimic_gaps) | np.isnan(eicu_gaps))
            print(f"\n=== model={model_type} window={window_h}h cross-task summary "
                  f"(n={valid.sum()} tasks) ===")
            if valid.sum() > 0:
                print(f"  mean MIMIC gap = {mimic_gaps[valid].mean():.4f}")
                print(f"  mean eICU gap  = {eicu_gaps[valid].mean():.4f}")
            if valid.sum() > 1:
                from scipy.stats import ttest_1samp
                t_m, p_m = ttest_1samp(mimic_gaps[valid], 0.0)
                t_e, p_e = ttest_1samp(eicu_gaps[valid], 0.0)
                print(f"  MIMIC gap vs 0: t={t_m:.3f} p={p_m:.4f}")
                print(f"  eICU  gap vs 0: t={t_e:.3f} p={p_e:.4f}")

            summary_path = (Path("results") / "exp5" / "summary" / model_type / seed_dir /
                             f"{window_h}h_odi_age_transfer_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(task_gaps, f, indent=2)
            print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
