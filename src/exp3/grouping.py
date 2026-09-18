"""Variable grouping sensitivity ablation: does ODI>0 survive alternative
assignments of the mixed (treatment/medication) variables to the disease or
observation feature groups?
"""

from __future__ import annotations

from ..data.feature_groups import DISEASE_FEATURES, OBSERVATION_FEATURES, MIXED_FEATURES
from ..data.transforms import prepare_xy
from ..models.train_utils import create_and_fit
from ..odi import compute_odi

# Variant A: mixed features assigned to D-Model
DISEASE_VARIANT_A = DISEASE_FEATURES + MIXED_FEATURES
OBSERVATION_VARIANT_A = OBSERVATION_FEATURES

# Variant B: mixed features assigned to O-Model
DISEASE_VARIANT_B = DISEASE_FEATURES
OBSERVATION_VARIANT_B = OBSERVATION_FEATURES + MIXED_FEATURES

VARIANTS = [
    ("original",  DISEASE_FEATURES,     OBSERVATION_FEATURES),
    ("variant_A", DISEASE_VARIANT_A,     OBSERVATION_VARIANT_A),
    ("variant_B", DISEASE_VARIANT_B,     OBSERVATION_VARIANT_B),
]


def run_variant(train_df, val_df, test_df, task: str, model_type: str, cfg: dict,
                 d_cols: list[str], o_cols: list[str]) -> dict:
    """Train disease-only and mixed models for one grouping variant, return ODI dict."""
    m_cols = list(dict.fromkeys(d_cols + o_cols + MIXED_FEATURES))

    X_tr_m, y_tr, tr_m = prepare_xy(train_df, m_cols, task, fit=True)
    X_te_m, y_te, _ = prepare_xy(test_df, m_cols, task, transformer=tr_m)
    X_val_m, y_val, _ = prepare_xy(val_df, m_cols, task, transformer=tr_m)

    X_tr_d, _, tr_d = prepare_xy(train_df, d_cols, task, fit=True)
    X_te_d, _, _ = prepare_xy(test_df, d_cols, task, transformer=tr_d)
    X_val_d, _, _ = prepare_xy(val_df, d_cols, task, transformer=tr_d)

    m_mixed = create_and_fit(model_type, cfg, X_tr_m, y_tr, X_val_m, y_val)
    m_disease = create_and_fit(model_type, cfg, X_tr_d, y_tr, X_val_d, y_val)

    return compute_odi(y_te, m_mixed.predict_proba(X_te_m), m_disease.predict_proba(X_te_d))
