"""Logistic Regression wrapper with unified sklearn-style interface."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class LogisticModel:
    """Thin wrapper around sklearn LogisticRegression.

    Exposes fit / predict_proba matching the interface used by EHRTrainer.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, solver: str = "lbfgs",
                 seed: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, solver=solver,
            class_weight="balanced", random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LogisticModel":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list[str] | None = None) -> dict:
        coef = self._model.coef_[0]
        if feature_names:
            return dict(zip(feature_names, coef.tolist()))
        return {"coef": coef.tolist()}
