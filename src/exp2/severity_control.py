"""Severity confounding control methods for experiment 2.

Three methods to verify ODI is not explained by severity differences:
  1. Propensity score matching on SOFA score
  2. Diagnosis-category stratification
  3. SOFA as additional covariate in O-Model
"""

from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    caliper: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """1:1 nearest-neighbour propensity score matching with caliper.

    Args:
        df: patient DataFrame
        treatment_col: binary column defining high-OII (1) vs low-OII (0) group
        covariates: columns to use for propensity score estimation
        caliper: maximum allowed propensity score difference (in std units)

    Returns:
        Matched subset DataFrame (roughly equal size high/low OII groups)
    """
    df = df.copy()
    X = df[covariates].fillna(df[covariates].median())
    y = df[treatment_col].values

    ps_model = LogisticRegression(max_iter=1000, random_state=seed)
    ps_model.fit(X, y)
    df["_ps"] = ps_model.predict_proba(X)[:, 1]

    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    # Caliper in standard deviation units
    ps_std = df["_ps"].std()
    caliper_abs = caliper * ps_std

    treated = treated.sample(frac=1, random_state=seed)

    matched_treated_idx = []
    matched_control_idx = []
    # Sorted scores avoid scanning the entire control cohort for every patient.
    # The original position breaks distance ties exactly as pandas.idxmin does.
    available = sorted((score, pos) for pos, score in enumerate(control["_ps"]))
    for treated_idx, score in treated["_ps"].items():
        if not available:
            break
        insertion = bisect_left(available, (score, -1))
        candidates = []
        if insertion < len(available):
            candidates.append(insertion)
        if insertion > 0:
            left_score = available[insertion - 1][0]
            candidates.append(bisect_left(available, (left_score, -1)))
        best = min(candidates, key=lambda i: (abs(available[i][0] - score), available[i][1]))
        if abs(available[best][0] - score) <= caliper_abs:
            _, control_pos = available.pop(best)
            matched_treated_idx.append(treated_idx)
            matched_control_idx.append(control.index[control_pos])

    matched_treated = treated.loc[matched_treated_idx]
    matched_control = control.loc[matched_control_idx]
    result = pd.concat([matched_treated, matched_control]).drop(columns=["_ps"])
    result = result.reset_index(drop=True)

    def balance(high, low):
        h, l = X.loc[high.index], X.loc[low.index]
        scale = np.sqrt((h.var() + l.var()) / 2)
        difference = h.mean() - l.mean()
        smd = difference / scale.replace(0, np.nan)
        smd = smd.mask((scale == 0) & (difference == 0), 0)
        return {c: float(v) if np.isfinite(v) else None for c, v in smd.items()}

    def score_summary(group):
        return {k: float(v) if np.isfinite(v) else None
                for k, v in group["_ps"].describe().items()}

    result.attrs["psm_diagnostics"] = {
        "seed": seed,
        "n_high": len(treated), "n_low": len(control),
        "n_pairs": len(matched_treated_idx),
        "n_unmatched_high": len(treated) - len(matched_treated_idx),
        "caliper_abs": float(caliper_abs),
        "smd_before": balance(treated, control),
        "smd_after": balance(matched_treated, matched_control),
        "propensity_before": {"high": score_summary(treated), "low": score_summary(control)},
        "propensity_after": {"high": score_summary(matched_treated), "low": score_summary(matched_control)},
    }
    return result


def stratify_by_diagnosis(
    df: pd.DataFrame,
    diagnosis_col: str = "dx_infection",
) -> dict[str, pd.DataFrame]:
    """Split cohort into diagnosis-category subsets.

    Uses the one-hot DX columns from feature_groups.

    Args:
        df: patient DataFrame with dx_* columns
        diagnosis_col: primary diagnosis column (unused; all dx_ cols are used)

    Returns:
        {category_name: sub_df}
    """
    dx_cols = [c for c in df.columns if c.startswith("dx_")]
    strata: dict[str, pd.DataFrame] = {}
    for col in dx_cols:
        cat = col.replace("dx_", "")
        sub = df[df[col] == 1].copy()
        if len(sub) >= 50:
            strata[cat] = sub
    # Patients with no DX match → "other"
    any_dx = df[dx_cols].any(axis=1)
    if (~any_dx).sum() >= 50:
        strata["unclassified"] = df[~any_dx].copy()
    return strata


def sofa_adjusted_analysis(
    X: np.ndarray,
    y: np.ndarray,
    sofa: np.ndarray,
    model_type: str,
    config: dict,
) -> dict:
    """Retrain O-Model with SOFA appended as extra covariate.

    Args:
        X: observation feature matrix
        y: labels
        sofa: SOFA scores (same length as X)
        model_type: "lr" | "xgb" | "gru" | "rf" | "lgbm" | "mlp"
        config: full training config, forwarded to create_and_fit

    Returns:
        metrics dict from the SOFA-adjusted model
    """
    from ..utils.metrics import compute_all_metrics
    from ..models.train_utils import create_and_fit

    sofa_col = sofa.reshape(-1, 1)
    X_aug = np.hstack([X, sofa_col])

    n = len(y)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)

    tr_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    te_idx = idx[n_train + n_val:]

    model = create_and_fit(
        model_type, config, X_aug[tr_idx], y[tr_idx], X_aug[val_idx], y[val_idx]
    )
    y_pred = model.predict_proba(X_aug[te_idx])
    return compute_all_metrics(y[te_idx], y_pred)
