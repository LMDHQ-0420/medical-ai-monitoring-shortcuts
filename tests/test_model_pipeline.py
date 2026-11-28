"""正式模型名单与预测身份的回归检查。"""
from src.exp1.trainer import MODELS
from src.models.train_utils import TABULAR_MODELS
from src.data.feature_groups import features_for


def test_active_model_set():
    assert MODELS == TABULAR_MODELS == ('lr', 'rf', 'xgb', 'lgbm', 'mlp')


def test_state_control_feature_counts():
    assert len(features_for('clinical_state_treatment')) == 20
    assert len(features_for('full_model')) == 50
    assert len(features_for('clinical_state_treatment','first_plus_latest')) == 26
    assert len(features_for('full_model','first_plus_latest')) == 56
