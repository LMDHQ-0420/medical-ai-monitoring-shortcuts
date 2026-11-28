import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

from .bootstrap import bootstrap_ci


def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_pred))


def brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_pred))


def fnr(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_bin = (y_pred >= threshold).astype(int)
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    return float(((y_bin[pos] == 0).sum()) / pos.sum())


def fpr(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_bin = (y_pred >= threshold).astype(int)
    neg = y_true == 0
    if neg.sum() == 0:
        return float("nan")
    return float(((y_bin[neg] == 1).sum()) / neg.sum())


def calibration_slope_intercept(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Logistic calibration: regress y_true on logit(y_pred)."""
    eps = 1e-7
    logit_pred = np.log(np.clip(y_pred, eps, 1 - eps) / (1 - np.clip(y_pred, eps, 1 - eps)))
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(
        logit_pred.reshape(-1, 1), y_true
    )
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    threshold: float = 0.5,
) -> dict:
    """Compute full metric suite with bootstrap CIs for AUROC."""
    auroc_val = auroc(y_true, y_pred)
    ci_lower, ci_upper = bootstrap_ci(auroc, y_true, y_pred, n_bootstrap=n_bootstrap)
    slope, intercept = calibration_slope_intercept(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "auroc": auroc_val,
        "auroc_ci_lower": ci_lower,
        "auroc_ci_upper": ci_upper,
        "auprc": auprc(y_true, y_pred),
        "brier": brier(y_true, y_pred),
        "calib_slope": slope,
        "calib_intercept": intercept,
        "fnr": fnr(y_true, y_pred, threshold),
        "fpr": fpr(y_true, y_pred, threshold),
    }
