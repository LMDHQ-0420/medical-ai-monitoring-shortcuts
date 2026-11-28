"""本轮时间边界、患者配对的回归检查。"""
import numpy as np
import pandas as pd
import pytest

from src.data.treatment_events import output_rows
from src.data.transforms import split_cohort


def test_patient_split_ignores_input_order():
    frame=pd.DataFrame({'subject_id':np.arange(40), 'stay_id':np.arange(40)+1000})
    a=split_cohort(frame,seed=42)
    b=split_cohort(frame.sample(frac=1,random_state=10),seed=42)
    for left,right in zip(a,b):
        assert set(left.subject_id)==set(right.subject_id)
    assert len(set(a[0].subject_id)&set(a[1].subject_id))==0
    assert len(set(a[0].subject_id)&set(a[2].subject_id))==0
    assert len(set(a[1].subject_id)&set(a[2].subject_id))==0


def test_treatment_after_cutoff_does_not_change_input():
    ids=np.array([1,2])
    idx=pd.MultiIndex.from_tuples([(1,1),(1,8),(2,2)],names=['stay_id','hour'])
    hourly=pd.DataFrame({'count':[1,1,1],'antibiotic':[0,1,1],'vaso':[0,1,0],'vent':[0,0,0]},index=idx)
    before=output_rows(ids,hourly.copy(),3)
    later=hourly.copy();later.loc[(1,8),'count']=99;later.loc[(1,8),'vaso']=0
    pd.testing.assert_frame_equal(before,output_rows(ids,later,3))
    assert before.loc[0,'antibiotics_24h']==0
    assert before.loc[1,'antibiotics_24h']==1


def test_semantic_predictor_sets_exclude_unavailable_fields():
    from src.data.feature_groups import (CLINICAL_STATE_FEATURES, OBSERVATION_FEATURES,
                                         TREATMENT_FEATURES, PREDICTOR_SETS, features_for)
    uncertain={'sofa_score','has_diabetes','has_ckd','has_heart_failure','has_copd','has_cancer',
               'dx_infection','dx_cardiovascular','dx_respiratory','dx_renal','dx_other'}
    assert uncertain.isdisjoint(CLINICAL_STATE_FEATURES)
    assert 'blood_culture_sent_24h' not in TREATMENT_FEATURES
    assert 'has_blood_culture_24h' in OBSERVATION_FEATURES
    assert len(features_for('clinical_state_treatment','first_plus_latest')) == 26
    assert len(features_for('full_model','first_plus_latest')) == 56
    assert set(PREDICTOR_SETS)=={'clinical_state','observation','clinical_state_treatment','full_model'}


def test_fit_identity_ignores_audit_column_but_not_training_value():
    from src.formal.identity import fit_identity
    df=pd.DataFrame({'subject_id':[1,2,3], 'stay_id':[11,12,13],
                     'icu_mortality':[0,1,0], 'hr_first':[70.,80.,90.], 'audit':[1,2,3]})
    args=(df,df,'icu_mortality',['hr_first'],'lr',42,{'C':1.0},'prep-v1','impl-v1',False)
    first=fit_identity(*args)
    assert first==fit_identity(df.assign(audit=99),*args[1:])
    assert first!=fit_identity(df.assign(hr_first=[71.,80.,90.]),*args[1:])
    assert first!=fit_identity(df.assign(icu_mortality=[1,1,0]),*args[1:])
    assert first!=fit_identity(df,df,'icu_mortality',['hr_first'],'xgb',42,{'C':1.0},'prep-v1','impl-v1',True)


def test_tabular_input_validation():
    from src.models.train_utils import create_and_fit
    x=np.zeros((4,2),dtype=np.float32);y=np.array([0,1,0,1])
    with pytest.raises(ValueError,match='不属于五种表格模型'):
        create_and_fit('unknown',{},x,y,x,y)
    with pytest.raises(ValueError,match='二维'):
        create_and_fit('lr',{},x[:,None,:],y,x,y)
