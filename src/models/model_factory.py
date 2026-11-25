"""Unified model factory — returns model instance based on config."""

from __future__ import annotations

from .logistic import LogisticModel
from .xgboost_model import XGBoostModel
from .random_forest import RandomForestModel
from .lightgbm_model import LightGBMModel
from .mlp_model import MLPModel


def create_model(model_type: str, config: dict, input_size: int | None = None):
    """Create and return a model instance.

    Args:
        model_type: "lr" | "xgb" | "rf" | "lgbm" | "mlp"
        config: full config dict (reads from config["training"][model_type])
        input_size: reserved for compatibility; tabular models do not use it

    Returns:
        Model instance with fit / predict_proba interface
    """
    if model_type == "lr":
        lr_cfg = config.get("training", {}).get("logistic", {})
        return LogisticModel(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver=lr_cfg.get("solver", "lbfgs"),
            seed=config.get("training", {}).get("seed", 42),
        )
    elif model_type == "xgb":
        xgb_cfg = config.get("training", {}).get("xgboost", {})
        return XGBoostModel(
            n_estimators=xgb_cfg.get("n_estimators", 500),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            eval_metric=xgb_cfg.get("eval_metric", "auc"),
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 20),
            seed=config.get("training", {}).get("seed", 42),
            n_jobs=xgb_cfg.get("n_jobs", 1),
        )
    elif model_type == "rf":
        rf_cfg = config.get("training", {}).get("random_forest", {})
        return RandomForestModel(
            n_estimators=rf_cfg.get("n_estimators", 500),
            max_depth=rf_cfg.get("max_depth", 10),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            n_jobs=rf_cfg.get("n_jobs", 1),
            seed=config.get("training", {}).get("seed", 42),
        )
    elif model_type == "lgbm":
        lgbm_cfg = config.get("training", {}).get("lightgbm", {})
        return LightGBMModel(
            n_estimators=lgbm_cfg.get("n_estimators", 500),
            max_depth=lgbm_cfg.get("max_depth", 6),
            learning_rate=lgbm_cfg.get("learning_rate", 0.05),
            subsample=lgbm_cfg.get("subsample", 0.8),
            colsample_bytree=lgbm_cfg.get("colsample_bytree", 0.8),
            early_stopping_rounds=lgbm_cfg.get("early_stopping_rounds", 20),
            seed=config.get("training", {}).get("seed", 42),
            n_jobs=lgbm_cfg.get("n_jobs", 1),
        )
    elif model_type == "mlp":
        mlp_cfg = config.get("training", {}).get("mlp", {})
        return MLPModel(
            hidden_layer_sizes=tuple(mlp_cfg.get("hidden_layer_sizes", [64, 32])),
            alpha=mlp_cfg.get("alpha", 0.0001),
            learning_rate_init=mlp_cfg.get("learning_rate_init", 0.001),
            max_iter=mlp_cfg.get("max_iter", 300),
            early_stopping=mlp_cfg.get("early_stopping", True),
            n_iter_no_change=mlp_cfg.get("n_iter_no_change", 10),
            seed=config.get("training", {}).get("seed", 42),
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Choose 'lr', 'xgb', 'rf', 'lgbm', or 'mlp'."
        )
