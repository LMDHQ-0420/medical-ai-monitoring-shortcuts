"""XGBoost wrapper with unified interface and feature importance output."""

from __future__ import annotations

import numpy as np
import xgboost as xgb


class XGBoostModel:
    """Thin wrapper around xgboost.XGBClassifier.

    Exposes fit / predict_proba matching the interface used by EHRTrainer.
    Supports early stopping via eval_set.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_metric: str = "auc",
        early_stopping_rounds: int = 20,
        seed: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self._model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            use_label_encoder=False,
            random_state=seed,
            verbosity=0,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostModel":
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list[str] | None = None) -> dict:
        imp = self._model.feature_importances_
        if feature_names:
            return dict(zip(feature_names, imp.tolist()))
        return {"importance": imp.tolist()}
