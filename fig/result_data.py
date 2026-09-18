"""Read existing result files only; never rewrite or fit experiment results."""

import json
from functools import lru_cache
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results"
TASKS = ["icu_mortality", "inhosp_mortality", "aki", "sepsis_deterioration", "escalation"]
TASK_LABELS = ["ICU Death", "Hospital Death", "AKI", "Sepsis", "Escalation"]
MODEL_KEYS = ["lr", "rf", "xgb", "lgbm", "gru", "mlp"]
MODEL_LABELS = ["LR", "RF", "XGBoost", "LightGBM", "GRU", "MLP"]
WINDOWS = [3, 6, 12, 18, 24]
SEEDS = [42, 43, 44, 45, 46]
AGES = ["<45", "45-65", "65-80", ">80"]
AGE_LABELS = ["≤45", "45–65", "65–80", ">80"]
KEYS = ["task", "model", "window"]


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw)


def read_csv(path):
    raw = path.read_bytes()
    from io import BytesIO
    return pd.read_csv(BytesIO(raw))


@lru_cache(None)
def exp1():
    records = []
    for task, model, window, seed in product(TASKS, MODEL_KEYS, WINDOWS, SEEDS):
        d = {"task": task, "model": model, "window": window, "seed": seed}
        for group in ["disease", "observation", "mixed"]:
            p = RESULTS / "exp1" / task / group / model / f"seed{seed}" / f"{window}h_metrics.json"
            d[group] = read_json(p)["auroc"]
        d["odi"] = d["mixed"] - d["disease"]
        records.append(d)
    return pd.DataFrame(records)


@lru_cache(None)
def exp2():
    records, strata = [], []
    for task, model, window, seed in product(TASKS, MODEL_KEYS, WINDOWS, SEEDS):
        p = RESULTS / "exp2" / task / "mixed" / model / f"seed{seed}" / f"{window}h_severity_control.json"
        j = read_json(p)
        key = dict(task=task, model=model, window=window, seed=seed)
        records.append({**key, **j["psm"]})
        for diagnosis, odi in j["stratification"].items():
            strata.append({**key, "diagnosis": diagnosis, "odi": odi})
    return pd.DataFrame(records), pd.DataFrame(strata)


@lru_cache(None)
def exp3():
    records = []
    for task, model, window, seed in product(TASKS, MODEL_KEYS, WINDOWS, SEEDS):
        p = RESULTS / "exp3" / task / "mixed" / model / f"seed{seed}" / f"{window}h_grouping_ablation.json"
        j = read_json(p)
        records.append(dict(task=task, model=model, window=window, seed=seed,
            original=j["original"]["odi"], residual=j["variant_A"]["odi"],
            mixed_change=j["variant_A"]["auroc_mixed"]-j["original"]["auroc_mixed"]))
    return pd.DataFrame(records)


@lru_cache(None)
def age():
    frames = []
    for task, model, window, seed in product(TASKS, MODEL_KEYS, WINDOWS, SEEDS):
        p = RESULTS / "exp4" / task / "mixed" / model / f"seed{seed}" / f"{window}h_odi_by_age_transfer.csv"
        frames.append(read_csv(p).assign(task=task, model=model, window=window, seed=seed))
    return pd.concat(frames, ignore_index=True)


def age_gaps():
    p = age().pivot(index=KEYS+["seed", "source"], columns="age_group", values="odi")
    return (p[">80"]-p["<45"]).rename("gap").reset_index().pivot(
        index=KEYS+["seed"], columns="source", values="gap").reset_index()


@lru_cache(None)
def hospitals():
    frames = []
    for task, model, window, seed in product(TASKS, MODEL_KEYS, WINDOWS, SEEDS):
        p = RESULTS / "exp5" / task / "mixed" / model / f"seed{seed}" / f"{window}h_by_hospital.csv"
        d = read_csv(p).rename(columns={"auroc_external": "auroc_mixed", "auroc_external_disease": "auroc_disease"})
        d = d[["hospital_id", "n_patients", "auroc_mixed", "auroc_disease", "odi_mimic", "odi_eicu", "odi_shrinkage"]]
        frames.append(d.assign(task=task, model=model, window=window, seed=seed))
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def combo(frame):
    return frame.groupby(KEYS, sort=False).mean(numeric_only=True).reset_index()
