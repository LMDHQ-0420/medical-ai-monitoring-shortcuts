"""Shared fit/predict dispatch for all model types, used by EHRTrainer and
standalone analysis scripts (04/11/12) so model-type handling lives in one place.
"""

from __future__ import annotations

import numpy as np

from .model_factory import create_model

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SKLEARN_STYLE = ("lr", "rf", "mlp")
EVAL_SET_STYLE = ("xgb", "lgbm")


def create_and_fit(
    model_type: str,
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    logger=None,
):
    """Create a model of model_type and fit it, returning an object exposing
    predict_proba(X) regardless of underlying implementation (sklearn-style,
    xgb/lgbm eval_set style, or GRU/torch).
    """
    if model_type == "gru":
        assert TORCH_AVAILABLE, "PyTorch not available for GRU training."
        seed = config.get("training", {}).get("seed", 42)
        torch.manual_seed(seed)
        gru = create_model("gru", config, input_size=X_train.shape[1])
        return _fit_gru(gru, config, X_train, y_train, X_val, y_val, logger, seed=seed)

    model = create_model(model_type, config)
    if model_type in EVAL_SET_STYLE:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)
    return model


def _fit_gru(model, config, X_train, y_train, X_val, y_val, logger=None, seed=42):
    gru_cfg = config["training"]["gru"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tr = torch.tensor(X_train[:, None, :], dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val[:, None, :], dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    shuffle_gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=gru_cfg["batch_size"], shuffle=True,
                         generator=shuffle_gen)
    optimizer = optim.Adam(model.parameters(), lr=gru_cfg["lr"])
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(gru_cfg["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= gru_cfg["patience"]:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _GRUPredictor(model)


class _GRUPredictor:
    """Wraps a fitted GRUModel so it exposes predict_proba(X) like sklearn."""

    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        device = next(self.model.parameters()).device
        X_t = torch.tensor(X[:, None, :], dtype=torch.float32).to(device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().squeeze(1)
        return preds
