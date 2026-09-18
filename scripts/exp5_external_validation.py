"""Experiment 5: Cross-hospital transfer failure (odi_shrinkage).

Trains mixed + disease models on MIMIC-IV, evaluates ODI (AUROC_mixed -
AUROC_disease) on the MIMIC-internal test set and on each eICU hospital, and
reports odi_shrinkage = odi_mimic - odi_eicu per hospital.

Iterates over all tasks x model_types x time_windows (aligned with exp1's grid
scale, though not all combinations may have data yet).

Usage:
    python scripts/exp5_external_validation.py --config configs/default.yaml
    python scripts/exp5_external_validation.py --task icu_mortality --model xgb --window 24
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd

from src.exp5.external_validation import fit_mixed_and_disease, evaluate_odi
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
        eicu_path = processed_dir / f"eicu_{window_h}h.parquet"
        if not mimic_path.exists() or not eicu_path.exists():
            print(f"[SKIP] window={window_h}h: parquet not found.")
            continue

        mimic_df = pd.read_parquet(mimic_path)
        eicu_df = pd.read_parquet(eicu_path)

        seed = cfg["training"]["seed"]
        ratios = tuple(cfg["training"]["train_val_test_split"])

        for model_type in model_types:
            pooled_results = []
            for task in tasks:
                if task not in mimic_df.columns or task not in eicu_df.columns:
                    print(f"[SKIP] Label column '{task}' not found in parquet.")
                    continue

                out_path = (Path("results") / "exp5" / task / "mixed" / model_type / seed_dir /
                            f"{window_h}h_by_hospital.csv")
                if out_path.exists():
                    print(f"[SKIP] {out_path} already exists.")
                    continue

                print(f"\n>>> task={task} | model={model_type} | window={window_h}h")
                task_df = mimic_df[mimic_df[task].notna()].copy()
                train_df, val_df, test_df = split_cohort(task_df, ratios=ratios, seed=seed)

                model_m, tr_m, model_d, tr_d = fit_mixed_and_disease(
                    train_df, val_df, task, model_type, cfg
                )
                internal = evaluate_odi(test_df, task, model_m, tr_m, model_d, tr_d)
                odi_mimic = internal["odi"]
                print(f"  Internal ODI={odi_mimic:.4f} "
                      f"(AUROC_mixed={internal['auroc_mixed']:.4f}, "
                      f"AUROC_disease={internal['auroc_disease']:.4f})")

                hospital_results = []
                for hosp_id, hosp_df in eicu_df.groupby("hospital_id"):
                    hosp_df = hosp_df[hosp_df[task].notna()]
                    if len(hosp_df) < 20:
                        continue
                    try:
                        ext = evaluate_odi(hosp_df, task, model_m, tr_m, model_d, tr_d)
                    except Exception as e:
                        print(f"  [SKIP] hospital {hosp_id}: {e}")
                        continue
                    hospital_results.append({
                        "hospital_id": str(hosp_id),
                        "n_patients": ext["n_patients"],
                        "auroc_mixed": ext["auroc_mixed"],
                        "auroc_disease": ext["auroc_disease"],
                        "odi_mimic": odi_mimic,
                        "odi_eicu": ext["odi"],
                        "odi_shrinkage": odi_mimic - ext["odi"],
                    })

                if not hospital_results:
                    print("  No eICU hospitals with sufficient data, skipping save.")
                    continue

                result_df = pd.DataFrame(hospital_results)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                result_df.to_csv(out_path, index=False)
                print(f"  Saved {len(result_df)} hospitals to {out_path}")

                shrink = result_df["odi_shrinkage"].dropna()
                if len(shrink) > 1:
                    from scipy.stats import ttest_1samp, wilcoxon
                    t_stat, t_p = ttest_1samp(shrink.values, popmean=0.0)
                    try:
                        w_stat, w_p = wilcoxon(shrink.values)
                    except ValueError:
                        w_stat, w_p = float("nan"), float("nan")
                    pct_positive = float((shrink > 0).mean())
                    print(f"  mean shrinkage={shrink.mean():.4f}, "
                          f"% hospitals with shrinkage>0 = {pct_positive:.1%}")
                    print(f"  paired t-test: t={t_stat:.3f} p={t_p:.4f}; Wilcoxon p={w_p:.4f}")

                result_df_pooled = result_df.copy()
                result_df_pooled["task"] = task
                pooled_results.append(result_df_pooled)

            if not pooled_results:
                continue
            pooled = pd.concat(pooled_results, ignore_index=True)
            pooled_path = (Path("results") / "exp5" / "pooled" / "mixed" / model_type / seed_dir /
                           f"{window_h}h_by_hospital.csv")
            pooled_path.parent.mkdir(parents=True, exist_ok=True)
            pooled.to_csv(pooled_path, index=False)
            print(f"\n  Pooled across {pooled['task'].nunique()} tasks, "
                  f"{len(pooled)} task-hospital rows saved to {pooled_path}")

            shrink_all = pooled["odi_shrinkage"].dropna()
            if len(shrink_all) > 1:
                from scipy.stats import ttest_1samp, wilcoxon
                t_stat, t_p = ttest_1samp(shrink_all.values, popmean=0.0)
                w_stat, w_p = wilcoxon(shrink_all.values)
                pct_positive = float((shrink_all > 0).mean())
                print(f"  pooled: mean shrinkage={shrink_all.mean():.4f}, "
                      f"% hospitals with shrinkage>0 = {pct_positive:.1%}")
                print(f"  paired t-test: t={t_stat:.3f} p={t_p:.4f}; Wilcoxon p={w_p:.4f}")


if __name__ == "__main__":
    main()
