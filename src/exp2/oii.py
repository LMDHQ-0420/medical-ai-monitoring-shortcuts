"""Observation Intensity Index (OII) computation.

OII = (1/K) * sum_k x̃_k
where x̃_k is the min-max normalized value of the k-th observation feature.
Normalization parameters are fit on the training set only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.feature_groups import OBSERVATION_FEATURES, get_obs_features_for_window



class OIIScaler:
    """Min-max scaler fit on training data for OII normalization."""

    def __init__(self) -> None:
        self._min: pd.Series | None = None
        self._max: pd.Series | None = None

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "OIIScaler":
        self._min = df[cols].min()
        self._max = df[cols].max()
        return self

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        assert self._min is not None, "Call fit before transform."
        denom = (self._max[cols] - self._min[cols]).replace(0, 1)
        return (df[cols] - self._min[cols]) / denom

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return self.fit(df, cols).transform(df, cols)


def compute_oii(
    df: pd.DataFrame,
    window_h: int,
    scaler: OIIScaler | None = None,
    weights: dict[str, float] | None = None,
    fit_scaler: bool = False,
) -> tuple[pd.Series, OIIScaler]:
    """Compute per-patient OII for a given time window.

    Args:
        df: DataFrame containing observation features
        window_h: time window in hours (6 / 12 / 24)
        scaler: pre-fit OIIScaler; if None and fit_scaler=True, a new one is created and fit
        weights: per-feature weights; None means equal weight
        fit_scaler: if True, fit the scaler on df (use only for training set)

    Returns:
        (oii_series, scaler) — oii_series has same index as df, values in [0, 1]
    """
    cols = _select_cols(df, window_h)
    if not cols:
        raise ValueError(f"No observation features found for window_h={window_h}")

    if scaler is None:
        scaler = OIIScaler()
        fit_scaler = True

    if fit_scaler:
        normalized = scaler.fit_transform(df, cols)
    else:
        normalized = scaler.transform(df, cols)

    normalized = normalized.clip(0, 1)  # clamp out-of-range values from test set

    if weights is not None:
        w = np.array([weights.get(c, 1.0) for c in cols], dtype=np.float64)
        w = w / w.sum()
        oii = normalized.values @ w
    else:
        oii = normalized.values.mean(axis=1)

    return pd.Series(oii, index=df.index, name=f"oii_{window_h}h"), scaler


def _select_cols(df: pd.DataFrame, window_h: int) -> list[str]:
    """Return observation feature columns present in df for the given window."""
    candidates = get_obs_features_for_window(window_h)
    # Also include window-agnostic observation features if present
    present = [c for c in candidates if c in df.columns]
    if not present:
        # Fall back to all observation features that exist in df
        present = [c for c in OBSERVATION_FEATURES if c in df.columns]
    return present
