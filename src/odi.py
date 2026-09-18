"""Observation Dependence Index (ODI) computation.

ODI = AUROC(M-Model) - AUROC(D-Model)

A positive ODI means the mixed model (which includes observation features)
outperforms the disease-only model, quantifying how much the model relies
on observation behaviour beyond disease severity.
"""

from __future__ import annotations

import numpy as np

from .utils.metrics import auroc
from .utils.bootstrap import bootstrap_ci


def compute_odi(
    y_true: np.ndarray,
    y_pred_mixed: np.ndarray,
    y_pred_disease: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict:
    """Compute ODI and bootstrap confidence interval.

    Args:
        y_true: true binary labels
        y_pred_mixed: predicted probabilities from the mixed model (D+O+M features)
        y_pred_disease: predicted probabilities from the disease-only model (D features)
        n_bootstrap: number of bootstrap iterations

    Returns:
        dict with keys: odi, ci_lower, ci_upper, auroc_mixed, auroc_disease
    """
    auroc_mixed = auroc(y_true, y_pred_mixed)
    auroc_disease = auroc(y_true, y_pred_disease)
    odi_point = auroc_mixed - auroc_disease

    def _odi(yt, ypm, ypd):
        am = auroc(yt, ypm)
        ad = auroc(yt, ypd)
        return am - ad

    ci_lower, ci_upper = bootstrap_ci(
        _odi, y_true, y_pred_mixed, y_pred_disease, n_bootstrap=n_bootstrap
    )

    return {
        "odi": float(odi_point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "auroc_mixed": float(auroc_mixed),
        "auroc_disease": float(auroc_disease),
    }
