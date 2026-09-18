"""EHR model trainer: handles all model types in a unified loop (see model_factory)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.feature_groups import DISEASE_FEATURES, OBSERVATION_FEATURES, MIXED_FEATURES
from ..data.transforms import EHRTransformer, split_cohort, prepare_xy
from ..models.train_utils import create_and_fit
from ..utils.metrics import compute_all_metrics
from ..utils.logger import get_logger, log_metrics


FEATURE_GROUP_MAP = {
    "disease":     DISEASE_FEATURES,
    "observation": OBSERVATION_FEATURES,
    "mixed":       DISEASE_FEATURES + OBSERVATION_FEATURES + MIXED_FEATURES,
}


class EHRTrainer:
    """Train, evaluate, and save results for one (task, feature_group, model_type, window) combo."""

    def __init__(
        self,
        model_type: str,
        task: str,
        feature_group: str,
        window_h: int,
        config: dict,
        output_base: str = "results",
        seed_dir: str | None = None,
    ) -> None:
        self.model_type = model_type
        self.task = task
        self.feature_group = feature_group
        self.window_h = window_h
        self.config = config
        self.output_base = Path(output_base)
        self.seed_dir = seed_dir or f"seed{config['training']['seed']}"
        self.logger = get_logger(f"{task}_{feature_group}_{model_type}_{window_h}h")

        self.feature_cols = FEATURE_GROUP_MAP[feature_group]
        self.model = None
        self.transformer: EHRTransformer | None = None
        self._X_test: np.ndarray | None = None
        self._y_test: np.ndarray | None = None
        self._test_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> dict:
        """Full pipeline: split → fit → evaluate → save.

        Args:
            df: cohort DataFrame with feature columns and label column

        Returns:
            metrics dict
        """
        ratios = tuple(self.config["training"]["train_val_test_split"])
        seed = self.config["training"]["seed"]

        # Filter to rows where label is non-null
        df = df[df[self.task].notna()].copy()

        train_df, val_df, test_df = split_cohort(df, ratios=ratios, seed=seed)
        self._test_df = test_df

        # Prepare features
        X_train, y_train, self.transformer = prepare_xy(
            train_df, self.feature_cols, self.task, fit=True
        )
        X_val, y_val, _ = prepare_xy(val_df, self.feature_cols, self.task,
                                      transformer=self.transformer)
        X_test, y_test, _ = prepare_xy(test_df, self.feature_cols, self.task,
                                        transformer=self.transformer)
        self._X_test = X_test
        self._y_test = y_test

        # Create and train model
        self.model = create_and_fit(
            self.model_type, self.config, X_train, y_train, X_val, y_val,
            logger=self.logger,
        )

        # Evaluate
        metrics = self.evaluate(X_test, y_test)
        self.save_results(metrics, test_df)
        return metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.model.predict_proba(X_test)
        n_bootstrap = self.config.get("analysis", {}).get("n_bootstrap", 1000)
        metrics = compute_all_metrics(y_test, y_pred, n_bootstrap=n_bootstrap)
        metrics["task"] = self.task
        metrics["feature_group"] = self.feature_group
        metrics["model_type"] = self.model_type
        metrics["window_h"] = self.window_h

        log_metrics(self.logger, {k: v for k, v in metrics.items()
                                  if isinstance(v, (int, float))},
                    step="test")
        return metrics

    def save_results(self, metrics: dict, test_df: pd.DataFrame) -> None:
        out_dir = self.output_base / self.task / self.feature_group / self.model_type / self.seed_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.window_h}h"

        # Save metrics JSON
        json_path = out_dir / f"{stem}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)
        self.logger.info(f"Metrics saved to {json_path}")

        # Save predictions CSV
        y_pred = self.model.predict_proba(self._X_test)
        pred_df = pd.DataFrame({
            "patient_id": test_df["subject_id"].values
                          if "subject_id" in test_df.columns else range(len(y_pred)),
            "stay_id": test_df["stay_id"].values
                       if "stay_id" in test_df.columns else range(len(y_pred)),
            "y_true": self._y_test,
            "y_pred": y_pred,
            "sofa": test_df["sofa_score"].values
                    if "sofa_score" in test_df.columns else np.nan,
            "split": "test",
        })
        csv_path = out_dir / f"{stem}_predictions.csv"
        pred_df.to_csv(csv_path, index=False)
        self.logger.info(f"Predictions saved to {csv_path}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
