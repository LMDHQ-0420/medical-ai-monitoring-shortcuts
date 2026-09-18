"""Cross-hospital transfer: does ODI (AUROC_mixed - AUROC_disease) survive
transfer from MIMIC-IV to individual eICU hospitals, or does it shrink?

Trains mixed + disease models once on MIMIC, evaluates on MIMIC-internal test
set and on each eICU hospital (no retraining), and reports
odi_shrinkage = odi_mimic - odi_eicu per hospital.
"""

from __future__ import annotations

from ..data.feature_groups import DISEASE_FEATURES, OBSERVATION_FEATURES, MIXED_FEATURES
from ..data.transforms import prepare_xy
from ..models.train_utils import create_and_fit
from ..utils.metrics import auroc

MIXED_FEATURE_COLS = DISEASE_FEATURES + OBSERVATION_FEATURES + MIXED_FEATURES


def fit_mixed_and_disease(train_df, val_df, task: str, model_type: str, cfg: dict):
    """Fit the mixed and disease models on MIMIC training data.

    Returns (model_mixed, transformer_mixed, model_disease, transformer_disease).
    """
    X_tr_m, y_tr, tr_m = prepare_xy(train_df, MIXED_FEATURE_COLS, task, fit=True)
    X_val_m, y_val, _ = prepare_xy(val_df, MIXED_FEATURE_COLS, task, transformer=tr_m)
    model_m = create_and_fit(model_type, cfg, X_tr_m, y_tr, X_val_m, y_val)

    X_tr_d, y_tr_d, tr_d = prepare_xy(train_df, DISEASE_FEATURES, task, fit=True)
    X_val_d, y_val_d, _ = prepare_xy(val_df, DISEASE_FEATURES, task, transformer=tr_d)
    model_d = create_and_fit(model_type, cfg, X_tr_d, y_tr_d, X_val_d, y_val_d)

    return model_m, tr_m, model_d, tr_d


def evaluate_odi(df, task: str, model_m, tr_m, model_d, tr_d) -> dict:
    """Compute AUROC(mixed), AUROC(disease), and ODI on a given dataframe split."""
    X_m, y_m, _ = prepare_xy(df, MIXED_FEATURE_COLS, task, transformer=tr_m)
    X_d, y_d, _ = prepare_xy(df, DISEASE_FEATURES, task, transformer=tr_d)
    auroc_mixed = auroc(y_m, model_m.predict_proba(X_m))
    auroc_disease = auroc(y_d, model_d.predict_proba(X_d))
    return {
        "n_patients": int(len(df)),
        "auroc_mixed": float(auroc_mixed),
        "auroc_disease": float(auroc_disease),
        "odi": float(auroc_mixed - auroc_disease),
    }
