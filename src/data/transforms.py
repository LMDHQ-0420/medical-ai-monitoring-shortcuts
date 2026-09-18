"""Feature standardization and imputation for EHR tabular and sequential data.

All scalers are fit on the training split only, then applied to val/test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .feature_groups import DISEASE_FEATURES, OBSERVATION_FEATURES, MIXED_FEATURES, LABEL_COLS


class EHRTransformer:
    """z-score normalization + median imputation for EHR features.

    Usage:
        t = EHRTransformer(feature_cols)
        X_train = t.fit_transform(train_df)
        X_val   = t.transform(val_df)
    """

    def __init__(self, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols
        self._scaler = StandardScaler()
        self._medians: pd.Series | None = None
        self._binary_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self._binary_cols = [
            c for c in self.feature_cols
            if df[c].dropna().isin([0, 1]).all()
        ]
        self._medians = df[self.feature_cols].median().fillna(0.0)
        filled = df[self.feature_cols].fillna(self._medians)
        # Binary cols stay as-is (already 0/1); numeric cols get z-scored
        numeric_cols = [c for c in self.feature_cols if c not in self._binary_cols]
        result = filled.copy()
        if numeric_cols:
            result[numeric_cols] = self._scaler.fit_transform(filled[numeric_cols])
        return result.values.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._medians is not None, "Call fit_transform before transform."
        filled = df[self.feature_cols].fillna(self._medians)
        numeric_cols = [c for c in self.feature_cols if c not in self._binary_cols]
        result = filled.copy()
        if numeric_cols:
            result[numeric_cols] = self._scaler.transform(filled[numeric_cols])
        return result.values.astype(np.float32)


def split_cohort(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Patient-level random split (no patient appears in two splits).

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    subjects = df["subject_id"].unique()
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_ids = set(subjects[:n_train])
    val_ids = set(subjects[n_train : n_train + n_val])
    test_ids = set(subjects[n_train + n_val :])

    train_df = df[df["subject_id"].isin(train_ids)].copy()
    val_df = df[df["subject_id"].isin(val_ids)].copy()
    test_df = df[df["subject_id"].isin(test_ids)].copy()
    return train_df, val_df, test_df


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    transformer: EHRTransformer | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray, EHRTransformer]:
    """Extract X and y arrays, applying or fitting the transformer.

    Args:
        df: cohort DataFrame
        feature_cols: list of feature column names
        label_col: target column name
        transformer: existing EHRTransformer (None to create new)
        fit: if True, call fit_transform; else call transform

    Returns:
        (X, y, transformer)
    """
    if transformer is None:
        transformer = EHRTransformer(feature_cols)
    if fit:
        X = transformer.fit_transform(df)
    else:
        X = transformer.transform(df)
    y = df[label_col].values.astype(np.float32)
    return X, y, transformer
