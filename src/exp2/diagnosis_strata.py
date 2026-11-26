"""复用冻结模型的 eICU 预测，另报截点前诊断录入分层。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
from src.exp1.trainer import MODELS, SEEDS, TASKS, WINDOWS
from src.exp5.external_validation import predict_external

STRATA = ("infection", "cardiovascular", "respiratory", "renal", "other_recorded", "not_recorded")


def run(run_dir: Path) -> None:
    diagnosis = pd.read_parquet(ROOT / "data/processed/eicu_fig2_diagnosis.parquet")
    rows = []
    for window in WINDOWS:
        eicu = pd.read_parquet(ROOT / f"data/processed/eicu_{window}h.parquet")
        groups = diagnosis.loc[diagnosis.window_h.eq(window)].drop(columns="window_h")
        for task in TASKS:
            for model in MODELS:
                for seed in SEEDS:
                    cell = run_dir / "models/tabular" / task / model / f"seed{seed}" / f"{window}h"
                    manifest = json.loads((cell / "manifest.json").read_text())
                    if manifest.get("status") != "complete" or manifest.get("run_type") != "formal":
                        raise ValueError(f"正式权重未完成：{cell}")
                    prediction = predict_external(cell, eicu.loc[eicu[task].notna()], task)
                    joined = prediction.merge(groups, on="stay_id", validate="one_to_one")
                    if len(joined) != len(prediction):
                        raise ValueError(f"eICU 诊断键不覆盖外部预测：{cell}")
                    for stratum in STRATA:
                        d = joined.loc[joined[stratum]]
                        n, positive = len(d), int(d.y_true.sum())
                        if n < 20 or positive == 0 or positive == n:
                            rows.append({"task": task, "model": model, "window_h": window, "seed": seed,
                                         "run_id": "formal", "diagnosis": stratum, "n": n,
                                         "positive": positive, "status": "ineligible",
                                         "reason": "n<20_or_single_label"})
                            continue
                        y = d.y_true.to_numpy(dtype=int)
                        aD = roc_auc_score(y, d.pred_D)
                        aDT = roc_auc_score(y, d.pred_DT)
                        aF = roc_auc_score(y, d.pred_F)
                        rows.append({"task": task, "model": model, "window_h": window, "seed": seed,
                                     "run_id": "formal", "diagnosis": stratum, "n": n,
                                     "positive": positive, "status": "complete", "reason": "",
                                     "auc_D": aD, "auc_DT": aDT, "auc_F": aF,
                                     "odi_raw": aF - aD, "odi_residual": aF - aDT,
                                     "definition": "diagnosis entered before prediction cutoff; named strata may overlap; no record is not no disease",
                                     "interpretation": "exploratory external comparison; outcome definitions not harmonized"})
                    print("eICU diagnosis", task, model, window, seed, flush=True)
    pd.DataFrame(rows).to_csv(run_dir / "exp2/diagnosis.csv.gz", index=False, compression="gzip")
    print("complete", len(rows), "rows", flush=True)
