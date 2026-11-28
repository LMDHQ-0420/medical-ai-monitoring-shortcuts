"""Fig. 2 截点、配对结构和匹配权限的回归检查。"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.exp2 import clinical_state, matching
from src.formal.statistics import paired_interval


def test_latest_available_vital_ignores_late_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(clinical_state, 'ROOT', tmp_path)
    directory = tmp_path / 'data/processed/cache/chartevents'
    directory.mkdir(parents=True)
    events = pd.DataFrame({
        'itemid': [220045, 220045, 220045],
        'charttime': ['2020-01-01 01:00:00', '2020-01-01 02:00:00', '2020-01-01 05:00:00'],
        'storetime': ['2020-01-01 04:00:00', '2020-01-01 02:05:00', '2020-01-01 05:05:00'],
        'valuenum': [99, 80, 70],
    })
    events.to_parquet(directory / '1.parquet', index=False)
    rows = {r['window_h']: r for r in clinical_state.extract_one((1, pd.Timestamp('2020-01-01')))}
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
    for name in matching.VITALS:
        frame[name+'_value'] = 1.
        frame[name+'_observed'] = True
    result = matching.pair(frame, .2, 5, 1., np.zeros(len(matching.CONT)),
                             np.ones(len(matching.CONT)))
    assert len(result) == 2
    assert set(result.high_stay_id) == {1, 2}
    assert set(result.low_stay_id) == {3, 4}
    assert not result.low_stay_id.duplicated().any()


def test_bootstrap_resamples_matched_pairs():
    y=np.array([1,0,1,0,1,0])
    prob={'clinical_state':np.array([.8,.2,.7,.3,.6,.4]),
          'clinical_state_treatment':np.array([.8,.2,.7,.3,.6,.4]),
          'full_model':np.array([.9,.1,.8,.2,.7,.3])}
    out=paired_interval(y,prob,30,'test-pairs',np.array([0,0,1,1,2,2]))
    assert out['bootstrap_unit']=='matched_pair'
    assert out['observation_gain_ci_low']<=out['observation_gain_ci_high']
