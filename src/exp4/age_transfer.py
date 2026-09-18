"""Age-stratified ODI: does the age-ODI gap (>80 vs <45) computed on MIMIC-IV
internal data survive transfer to pooled eICU?
"""

from __future__ import annotations

import pandas as pd

from ..data.feature_groups import DISEASE_FEATURES, OBSERVATION_FEATURES, MIXED_FEATURES
from ..data.transforms import prepare_xy
from ..utils.metrics import auroc

MIXED_FEATURE_COLS = DISEASE_FEATURES + OBSERVATION_FEATURES + MIXED_FEATURES
AGE_BINS = [0, 45, 65, 80, 120]
AGE_LABELS = ["<45", "45-65", "65-80", ">80"]


def odi_by_age(df: pd.DataFrame, task: str, model_m, model_d, tr_m, tr_d) -> pd.DataFrame:
    """Compute AUROC(disease), AUROC(mixed), and ODI within each age group."""
    df = df[df[task].notna()].copy()
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    records = []
    for level, sub in df.groupby("age_group", observed=True):
        if sub[task].nunique() < 2 or len(sub) < 20:
            continue
        X_m, y_m, _ = prepare_xy(sub, MIXED_FEATURE_COLS, task, transformer=tr_m)
        X_d, y_d, _ = prepare_xy(sub, DISEASE_FEATURES, task, transformer=tr_d)
        auroc_m = auroc(y_m, model_m.predict_proba(X_m))
        auroc_d = auroc(y_d, model_d.predict_proba(X_d))
        records.append({
            "age_group": level,
            "n": int(len(sub)),
            "auroc_disease": auroc_d,
            "auroc_mixed": auroc_m,
            "odi": auroc_m - auroc_d,
        })
    return pd.DataFrame(records)


def age_gap(df: pd.DataFrame) -> float:
    """mimic_gap = ODI(>80) - ODI(<45); nan if either group is missing."""
    if "<45" not in df["age_group"].values or ">80" not in df["age_group"].values:
        return float("nan")
    return float(df.set_index("age_group").loc[">80", "odi"] -
                  df.set_index("age_group").loc["<45", "odi"])
