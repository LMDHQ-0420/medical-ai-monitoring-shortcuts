"""MLP (feed-forward neural network) wrapper with unified sklearn-style interface."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class MLPModel:
    """Thin wrapper around sklearn MLPClassifier.

    Exposes fit / predict_proba matching the interface used by EHRTrainer.
    Features are standardized internally since MLP is scale-sensitive.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 300,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        seed: int = 42,
    ) -> None:
        self._scaler = StandardScaler()
        self._model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MLPModel":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]
