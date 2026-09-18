"""Experiment 2: Severity control ablation.

Tests whether ODI remains after controlling for disease severity via:
  1. Propensity score matching on SOFA
  2. Diagnosis-category stratification
  3. SOFA as additional covariate in O-Model

Iterates over all tasks x model_types x time_windows (aligned with exp1's grid
scale, though not all combinations may have data yet).

Usage:
    python scripts/exp2_ablation_severity.py --config configs/default.yaml
    python scripts/exp2_ablation_severity.py --task icu_mortality --model xgb --window 24
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd

from src.exp2.severity_control import (
    propensity_score_matching,
    stratify_by_diagnosis,
    sofa_adjusted_analysis,
)
from src.odi import compute_odi
from src.exp2.oii import compute_oii
from src.data.transforms import split_cohort, prepare_xy
from src.data.feature_groups import DISEASE_FEATURES, OBSERVATION_FEATURES, MIXED_FEATURES
from src.models.train_utils import create_and_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default=None, help="Run single task only")
    parser.add_argument("--model", default=None, choices=["lr", "xgb", "gru", "rf", "lgbm", "mlp"],
                        help="Run single model only")
    parser.add_argument("--window", type=int, default=None, help="Run single window only")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config training.seed; also selects the seedN/ output subdir")
    parser.add_argument("--psm-only", action="store_true",
                        help="Overwrite PSM results only, preserving other existing analyses")
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

                out_path = (Path("results") / "exp2" / task / "mixed" / model_type / seed_dir /
                            f"{window_h}h_severity_control.json")

                print(f"\n>>> task={task} | model={model_type} | window={window_h}h")
                task_df = df[df[task].notna()].copy()
                train_df, val_df, test_df = split_cohort(task_df, ratios=ratios, seed=seed)
                _, oii_scaler = compute_oii(train_df, window_h, fit_scaler=True)

                task_results = {}
                if args.psm_only and out_path.exists():
                    with open(out_path) as f:
                        task_results = json.load(f)

                # --- Method 1: Propensity score matching ---
                oii_all, _ = compute_oii(task_df, window_h, scaler=oii_scaler)
                task_df["_oii"] = oii_all.values
                median_oii = task_df["_oii"].median()
                task_df["_high_oii"] = (task_df["_oii"] > median_oii).astype(int)

                covariates = [c for c in DISEASE_FEATURES if c in task_df.columns]
                try:
                    matched_df = propensity_score_matching(task_df, "_high_oii", covariates, seed=seed)
                    tr_m, va_m, te_m = split_cohort(matched_df, ratios=ratios, seed=seed)
                    odi_psm = _compute_odi_on_split(tr_m, va_m, te_m, task, model_type, cfg)
                    odi_psm["diagnostics"] = matched_df.attrs["psm_diagnostics"]
                    odi_psm["diagnostics"]["split_rows"] = {
                        "train": len(tr_m), "val": len(va_m), "test": len(te_m),
                    }
                    task_results["psm"] = odi_psm
                    print(f"  PSM: ODI={odi_psm['odi']:.4f}")
                except Exception as e:
                    if args.psm_only:
                        raise
                    print(f"  PSM failed: {e}")

                # --- Method 2: Diagnosis stratification ---
                strata = {} if args.psm_only else stratify_by_diagnosis(task_df)
                strat_odi = {}
                for cat, sub_df in strata.items():
                    try:
                        tr_s, va_s, te_s = split_cohort(sub_df, ratios=ratios, seed=seed)
                        odi_s = _compute_odi_on_split(tr_s, va_s, te_s, task, model_type, cfg)
                        strat_odi[cat] = odi_s["odi"]
                        print(f"  Stratum {cat}: ODI={odi_s['odi']:.4f}")
                    except Exception as e:
                        print(f"  Stratum {cat} failed: {e}")
                if not args.psm_only:
                    task_results["stratification"] = strat_odi

                # --- Method 3: SOFA-adjusted O-Model ---
                if not args.psm_only and "sofa_score" in test_df.columns:
                    try:
                        X_o_tr, y_tr, _ = prepare_xy(train_df, OBSERVATION_FEATURES, task, fit=True)
                        sofa_tr = train_df["sofa_score"].fillna(0).values
                        odi_sofa = sofa_adjusted_analysis(X_o_tr, y_tr, sofa_tr, model_type, cfg)
                        task_results["sofa_adjusted"] = odi_sofa
                        print(f"  SOFA-adjusted O-Model AUROC={odi_sofa.get('auroc', float('nan')):.4f}")
                    except Exception as e:
                        print(f"  SOFA-adjusted failed: {e}")

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(task_results, f, indent=2, default=str)
                print(f"  Saved to {out_path}")


def _compute_odi_on_split(train_df, val_df, test_df, task, model_type, cfg):
    feat_m = DISEASE_FEATURES + OBSERVATION_FEATURES + MIXED_FEATURES
    feat_d = DISEASE_FEATURES

    X_tr_m, y_tr, tr_m = prepare_xy(train_df, feat_m, task, fit=True)
    X_te_m, y_te, _ = prepare_xy(test_df, feat_m, task, transformer=tr_m)
    X_v_m, y_v, _ = prepare_xy(val_df, feat_m, task, transformer=tr_m)

    X_tr_d, _, tr_d = prepare_xy(train_df, feat_d, task, fit=True)
    X_te_d, _, _ = prepare_xy(test_df, feat_d, task, transformer=tr_d)
    X_v_d, _, _ = prepare_xy(val_df, feat_d, task, transformer=tr_d)

    m_mixed = create_and_fit(model_type, cfg, X_tr_m, y_tr, X_v_m, y_v)
    m_disease = create_and_fit(model_type, cfg, X_tr_d, y_tr, X_v_d, y_v)

    return compute_odi(y_te, m_mixed.predict_proba(X_te_m), m_disease.predict_proba(X_te_d))


if __name__ == "__main__":
    main()
