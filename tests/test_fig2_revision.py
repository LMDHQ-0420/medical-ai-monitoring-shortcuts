"""Fig. 2 截点、配对结构和匹配权限的回归检查。"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pnas_revision import build_fig2_state, fig2_evaluate, fig2_match


def test_latest_available_vital_ignores_late_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(build_fig2_state, 'ROOT', tmp_path)
    directory = tmp_path / 'data/processed/cache/chartevents'
    directory.mkdir(parents=True)
    events = pd.DataFrame({
        'itemid': [220045, 220045, 220045],
        'charttime': ['2020-01-01 01:00:00', '2020-01-01 02:00:00', '2020-01-01 05:00:00'],
        'storetime': ['2020-01-01 04:00:00', '2020-01-01 02:05:00', '2020-01-01 05:05:00'],
        'valuenum': [99, 80, 70],
    })
    events.to_parquet(directory / '1.parquet', index=False)
    rows = {r['window_h']: r for r in build_fig2_state.extract_one((1, pd.Timestamp('2020-01-01')))}
    assert rows[3]['hr_value'] == 80
    assert rows[3]['hr_charttime'] == pd.Timestamp('2020-01-01 02:00:00')
    assert rows[3]['hr_available_time'] == pd.Timestamp('2020-01-01 02:05:00')
    assert rows[6]['hr_value'] == 70


def test_matching_uses_exact_context_and_unique_patients():
    high = pd.DataFrame({'stay_id': [1, 2], 'subject_id': [11, 12], 'high_oii': True,
                         'logit_ps': [0., 0.1], 'age': [50., 60.],
                         'gender': [0, 1], 'admission_source': [1, 2]})
    low = pd.DataFrame({'stay_id': [3, 4], 'subject_id': [13, 14], 'high_oii': False,
                        'logit_ps': [0.02, 0.12], 'age': [51., 61.],
                        'gender': [0, 1], 'admission_source': [1, 2]})
    frame = pd.concat([high, low], ignore_index=True)
    frame['oii'] = [.8, .9, .2, .1]
    for name in fig2_match.VITALS:
        frame[name+'_value'] = 1.
        frame[name+'_observed'] = True
    result = fig2_match.pair(frame, .2, 5, 1., np.zeros(len(fig2_match.CONT)),
                             np.ones(len(fig2_match.CONT)))
    assert len(result) == 2
    assert set(result.high_stay_id) == {1, 2}
    assert set(result.low_stay_id) == {3, 4}
    assert not result.low_stay_id.duplicated().any()


def test_bootstrap_resamples_matched_pairs():
    frame = pd.DataFrame({
        'pair_index': [0, 0, 1, 1, 2, 2], 'arm': ['high', 'low']*3,
        'y_true': [1, 0, 1, 0, 1, 0],
        'pred_D': [.8, .2, .7, .3, .6, .4],
        'pred_DT': [.8, .2, .7, .3, .6, .4],
        'pred_F': [.9, .1, .8, .2, .7, .3],
    })
    out = fig2_evaluate.pair_interval(frame, 42, 30)
    assert out['bootstrap_unit'] == 'matched_pair'
    assert out['odi_residual_ci_low'] <= out['odi_residual_ci_high']
