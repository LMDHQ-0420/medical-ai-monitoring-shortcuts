"""Random Forest wrapper with unified sklearn-style interface."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    """Thin wrapper around sklearn RandomForestClassifier.

    Exposes fit / predict_proba matching the interface used by EHRTrainer.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            class_weight="balanced",
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RandomForestModel":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list[str] | None = None) -> dict:
        imp = self._model.feature_importances_
        if feature_names:
            return dict(zip(feature_names, imp.tolist()))
        return {"importance": imp.tolist()}
