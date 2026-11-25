"""独立的匹配训练敏感性分析：24h、LR、五任务五种子，不替代冻结模型 Fig. 2。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.pnas_revision.fig2_match import CONT, MASK, VITALS, pair, smd
from scripts.pnas_revision.fig2_evaluate import pair_interval
from src.data.transforms import EHRTransformer, split_cohort
from src.data.feature_groups import DISEASE_FEATURES, OBSERVATION_FEATURES, MIXED_FEATURES
from src.exp2.oii import OIIScaler
from src.models.train_utils import create_and_fit

GROUPS = {"D": DISEASE_FEATURES, "DT": DISEASE_FEATURES + MIXED_FEATURES,
          "F": DISEASE_FEATURES + OBSERVATION_FEATURES + MIXED_FEATURES}
TASKS = ("icu_mortality", "inhosp_mortality", "aki", "sepsis_deterioration", "escalation")


def run(run_dir: Path) -> None:
    state = pd.read_parquet(ROOT / "data/processed/mimic_iv_fig2_clinical_state.parquet")
    cohort = pd.read_parquet(ROOT / "data/processed/mimic_iv_24h.parquet")
    state = state.loc[state.window_h.eq(24)].drop(columns="window_h")
    cohort = cohort.merge(state, on="stay_id", validate="one_to_one")
    test_pairs = pd.read_csv(run_dir / "fig2_matching/matched_pairs.csv.gz")
    result = []
    out = run_dir / "fig2_refit_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44, 45, 46):
        train, val, test = split_cohort(cohort, seed=seed)
        train = train.copy()
        for vital in VITALS:
            column = vital + "_value"
            lower, upper = train[column].quantile([.01, .99]).to_numpy(dtype=float)
            train[column] = train[column].clip(lower, upper)
        scaler = OIIScaler().fit(train, OBSERVATION_FEATURES)
        train["oii"] = scaler.transform(train, OBSERVATION_FEATURES).clip(0, 1).mean(axis=1).to_numpy()
        threshold = float(train.oii.median())
        train["high_oii"] = train.oii.gt(threshold)
        cols = list(CONT + MASK + ("gender", "admission_source"))
        x = train[cols].astype(float)
        propensity = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(max_iter=1000, random_state=seed))
        propensity.fit(x, train.high_oii.astype(int))
        score = np.clip(propensity.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        train["logit_ps"] = np.log(score / (1 - score))
        logit_sd = float(train.logit_ps.std())
        numeric = train[list(CONT)].to_numpy(dtype=float)
        median, spread = np.nanmedian(numeric, axis=0), np.nanstd(numeric, axis=0)
        median[~np.isfinite(median)] = 0
        spread[~np.isfinite(spread) | (spread < 1e-6)] = 1
        training_pairs = pair(train, .2, 10, logit_sd, median, spread)
        chosen_ids = pd.concat([training_pairs.high_stay_id, training_pairs.low_stay_id]).to_numpy()
        matched_train = train.loc[train.stay_id.isin(chosen_ids)].copy()
        balanced = smd(train, train.stay_id.isin(chosen_ids).to_numpy())
        pd.DataFrame(balanced).to_csv(out / f"train_balance_seed{seed}.csv", index=False)
        training_pairs.to_csv(out / f"train_pairs_seed{seed}.csv.gz", index=False, compression="gzip")
        p = test_pairs.loc[test_pairs.window_h.eq(24) & test_pairs.seed.eq(seed)]
        ids = pd.concat([p.high_stay_id, p.low_stay_id])
        test_subset = test.loc[test.stay_id.isin(ids)].copy()
        pair_map = pd.concat([
            p[["pair_index", "high_stay_id"]].rename(columns={"high_stay_id": "stay_id"}).assign(arm="high"),
            p[["pair_index", "low_stay_id"]].rename(columns={"low_stay_id": "stay_id"}).assign(arm="low"),
        ], ignore_index=True)
        cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        cfg["training"]["seed"] = seed
        for task in TASKS:
            pred = test_subset[["subject_id", "stay_id", task]].rename(
                columns={"subject_id": "pair_id", task: "y_true"}).copy()
            for group, features in GROUPS.items():
                transformer = EHRTransformer(features)
                x_train = transformer.fit_transform(matched_train)
                x_val = transformer.transform(val)
                x_test = transformer.transform(test_subset)
                model = create_and_fit("lr", cfg, x_train, matched_train[task].to_numpy(),
                                       x_val, val[task].to_numpy())
                pred[f"pred_{group}"] = model.predict_proba(x_test)
                cell = out / task / f"seed{seed}"
                cell.mkdir(parents=True, exist_ok=True)
                joblib.dump(transformer, cell / f"transformer_{group}.joblib")
                joblib.dump(model, cell / f"model_{group}.joblib")
            pred = pred.merge(pair_map, on="stay_id", validate="one_to_one")
            dest = out / task / f"seed{seed}" / "24h_predictions.csv.gz"
            pred.to_csv(dest, index=False, compression="gzip", float_format="%.17g")
            y = pred.y_true.to_numpy(dtype=int)
            aD, aDT, aF = [roc_auc_score(y, pred[f"pred_{g}"].to_numpy()) for g in ("D", "DT", "F")]
            ci = pair_interval(pred, seed + TASKS.index(task) * 1000, 200)
            result.append({"task": task, "model": "lr", "window_h": 24, "seed": seed,
                           "analysis": "refit_on_matched_training_cohort", "n_train_matched": len(matched_train),
                           "n_train_pairs": len(training_pairs), "n_test_matched": len(pred),
                           "positive_test": int(y.sum()), "auc_D": aD, "auc_DT": aDT, "auc_F": aF,
                           "odi_raw": aF - aD, "odi_residual": aF - aDT,
                           "prediction_file": str(dest.relative_to(ROOT)), **ci})
            print("refit", task, seed, len(matched_train), len(pred), flush=True)
    pd.DataFrame(result).to_csv(run_dir / "matched_refit_sensitivity.csv.gz", index=False, compression="gzip")
    (out / "scope.json").write_text(json.dumps({
        "model": "lr", "window_h": 24, "tasks": list(TASKS), "seeds": [42,43,44,45,46],
        "match_training": "train-only, fixed logit caliper 0.2 train SD, age limit 10 years, exact sex/admission source",
        "validation": "original validation split", "evaluation": "same test-only matched pairs as frozen-model Fig. 2",
        "interpretation": "separate sensitivity, not mixed into primary matched metrics"
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    run(ap.parse_args().run_dir.resolve())
