"""冻结正式模型，在测试集匹配对子集与入院情境中重算预测增益。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
from src.exp1.trainer import GROUPS, MODELS, SEEDS, TASKS, WINDOWS, digest_file

CONTEXT = {0: "Other/unknown", 1: "Emergency", 2: "Transfer", 3: "Elective/procedure"}


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def pair_interval(frame: pd.DataFrame, seed: int, n_bootstrap: int) -> dict:
    ordered = frame.sort_values(["pair_index", "arm"])
    if len(ordered) != 2 * ordered.pair_index.nunique():
        raise ValueError("每个匹配对子必须有两名患者")
    values = ordered[["y_true", "pred_D", "pred_DT", "pred_F"]].to_numpy().reshape(-1, 2, 4)
    rng = np.random.default_rng(seed)
    raw, residual = [], []
    for _ in range(n_bootstrap):
        draw = values[rng.integers(0, len(values), len(values))].reshape(-1, 4)
        y = draw[:, 0]
        if y.min() == y.max():
            continue
        f = auc(y, draw[:, 3])
        raw.append(f - auc(y, draw[:, 1]))
        residual.append(f - auc(y, draw[:, 2]))
    if len(raw) < n_bootstrap * .95:
        raise ValueError("匹配对子自助抽样有效次数不足")
    return {
        "n_bootstrap": n_bootstrap,
        "bootstrap_unit": "matched_pair",
        "bootstrap_method": "percentile_2.5_97.5",
        "odi_raw_ci_low": float(np.quantile(raw, .025)),
        "odi_raw_ci_high": float(np.quantile(raw, .975)),
        "odi_residual_ci_low": float(np.quantile(residual, .025)),
        "odi_residual_ci_high": float(np.quantile(residual, .975)),
    }


def run(run_dir: Path, n_bootstrap: int, only_task: str | None = None) -> None:
    pairs = pd.read_csv(run_dir / "exp2/matched_pairs.csv.gz")
    original = pd.read_csv(run_dir / "exp1/results.csv.gz")
    matched, context = [], []
    for window in WINDOWS:
        cohort = pd.read_parquet(ROOT / f"data/processed/mimic_iv_{window}h.parquet",
                                 columns=["stay_id", "subject_id", "admission_source", *TASKS])
        for seed in SEEDS:
            p = pairs.loc[pairs.window_h.eq(window) & pairs.seed.eq(seed)]
            high = p[["pair_index", "high_stay_id", "high_pair_id"]].rename(
                columns={"high_stay_id": "stay_id", "high_pair_id": "pair_id"})
            low = p[["pair_index", "low_stay_id", "low_pair_id"]].rename(
                columns={"low_stay_id": "stay_id", "low_pair_id": "pair_id"})
            high["arm"] = "high"; low["arm"] = "low"
            selected = pd.concat([high, low], ignore_index=True)
            if len(selected) != 2 * len(p) or selected.stay_id.duplicated().any():
                raise ValueError("匹配关系不成对或重复")
            for task in (TASKS if only_task is None else (only_task,)):
                for model in MODELS:
                    cell = run_dir / "models/tabular" / task / model / f"seed{seed}" / f"{window}h"
                    m = json.loads((cell / "manifest.json").read_text())
                    predpath = cell / "predictions.csv.gz"
                    if (m.get("status") != "complete" or m.get("run_type") != "formal"
                        or m.get("feature_columns") != GROUPS
                        or digest_file(predpath) != m.get("predictions_sha256")):
                        raise ValueError(f"固定预测文件校验失败：{predpath}")
                    pred = pd.read_csv(predpath)
                    expect = cohort[["stay_id", "subject_id", task]].rename(columns={"subject_id": "pair_id", task: "label"})
                    labels = pred[["stay_id", "pair_id", "y_true"]].merge(
                        expect, on=["stay_id", "pair_id"], how="left", validate="one_to_one")
                    if labels.label.isna().any() or not np.array_equal(labels.y_true.to_numpy(), labels.label.to_numpy()):
                        raise ValueError("固定预测与处理队列的患者或标签不符")
                    sub = selected.merge(pred, on=["stay_id", "pair_id"], how="left", validate="one_to_one")
                    if sub.y_true.isna().any() or len(sub) != 2 * len(p):
                        raise ValueError("匹配患者与正式预测不一致")
                    out = run_dir / "exp2/predictions" / task / model / f"seed{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    dest = out / f"{window}h.csv.gz"
                    temp = dest.with_suffix(".tmp.csv.gz")
                    sub.to_csv(temp, index=False, compression="gzip", float_format="%.17g")
                    temp.replace(dest)
                    y = sub.y_true.to_numpy(dtype=int)
                    aD, aDT, aF = (auc(y, sub[f"pred_{g}"].to_numpy()) for g in ("D", "DT", "F"))
                    s = f"{task}/{model}/{window}/{seed}".encode()
                    bseed = int.from_bytes(hashlib.sha256(s).digest()[:8], "little")
                    ci = pair_interval(sub, bseed, n_bootstrap)
                    old = original.loc[(original.task == task) & (original.model == model)
                                       & (original.window_h == window) & (original.seed == seed)].iloc[0]
                    if abs(float(old.odi_raw) - (m["auc_F"] - m["auc_D"])) > 1e-10:
                        raise ValueError("原测试集 ODI 与正式 manifest 不一致")
                    matched.append({"task": task, "model": model, "window_h": window, "seed": seed,
                                    "run_id": "formal", "n_original": len(pred), "positive_original": int(pred.y_true.sum()),
                                    "n_pairs": len(p), "n_matched": len(sub), "positive_matched": int(y.sum()),
                                    "n_unmatched": len(pred) - len(sub),
                                    "auc_D_original": float(old.auc_D), "auc_DT_original": float(old.auc_DT),
                                    "auc_F_original": float(old.auc_F),
                                    "odi_raw_original": float(old.odi_raw), "odi_residual_original": float(old.odi_residual),
                                    "auc_D_matched": aD, "auc_DT_matched": aDT, "auc_F_matched": aF,
                                    "odi_raw_matched": aF - aD, "odi_residual_matched": aF - aDT,
                                    "prediction_file": str(dest.relative_to(ROOT)), "prediction_sha256": digest_file(dest), **ci})
                    merged = pred.merge(cohort[["stay_id", "admission_source"]], on="stay_id", validate="one_to_one")
                    for code, group in merged.groupby("admission_source"):
                        y0 = group.y_true.to_numpy(dtype=int)
                        if len(group) < 20 or len(np.unique(y0)) < 2:
                            continue
                        d, dt, f = (auc(y0, group[f"pred_{g}"].to_numpy()) for g in ("D", "DT", "F"))
                        context.append({"task": task, "model": model, "window_h": window, "seed": seed,
                                        "run_id": "formal", "stratum": CONTEXT.get(int(code), str(code)),
                                        "source_code": int(code), "n": len(group), "positive": int(y0.sum()),
                                        "auc_D": d, "auc_DT": dt, "auc_F": f,
                                        "odi_raw": f - d, "odi_residual": f - dt,
                                        "stratum_definition": "admission_source recorded at ICU admission; not final diagnosis"})
                    print("evaluated", task, model, window, seed, len(p), flush=True)
    suffix = "" if only_task is None else "_" + only_task
    pd.DataFrame(matched).to_csv(run_dir / "exp2" / ("results.csv.gz" if not suffix else f"results{suffix}.csv.gz"), index=False, compression="gzip")
    pd.DataFrame(context).to_csv(run_dir / "exp2" / ("context.csv.gz" if not suffix else f"context{suffix}.csv.gz"), index=False, compression="gzip")
    print("complete", len(matched), "matched cells", len(context), "context rows")
