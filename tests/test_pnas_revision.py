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


def test_treatment_group_excludes_observation_order():
    from src.data.feature_groups import MIXED_FEATURES,OBSERVATION_FEATURES
    assert 'blood_culture_sent_24h' not in MIXED_FEATURES
    assert 'has_blood_culture_24h' in OBSERVATION_FEATURES
    assert set(MIXED_FEATURES)=={'antibiotics_24h','vasopressors_24h','mechanical_ventilation_24h'}


def test_active_d_excludes_whole_stay_fields():
    from src.data.feature_groups import DISEASE_FEATURES
    uncertain={'sofa_score','has_diabetes','has_ckd','has_heart_failure','has_copd','has_cancer',
               'dx_infection','dx_cardiovascular','dx_respiratory','dx_renal','dx_other'}
    assert uncertain.isdisjoint(DISEASE_FEATURES)
    assert {'age','gender','admission_source','temp_first','creatinine_first'}.issubset(DISEASE_FEATURES)


def test_smoke_and_formal_paths_and_wrong_sample_cache():
    from pathlib import Path
    from src.exp1.trainer import cell_path,paired_cache_matches
    root=Path('/tmp/pnas_test')
    smoke=cell_path(root,'icu_mortality','lr',3,42,True)
    formal=cell_path(root,'icu_mortality','lr',3,42,False)
    assert smoke!=formal and '/smoke/' in str(smoke) and '/models/' in str(formal)
    expected=pd.DataFrame({'subject_id':[1,2,3],'stay_id':[11,12,13],'icu_mortality':[0,1,0]})
    saved=pd.DataFrame({'pair_id':[1,2,3],'stay_id':[11,12,13],'y_true':[0,1,0]})
    manifest={'split_counts':{'test':{'n':3,'positive':1}}}
    assert paired_cache_matches(saved,expected,manifest,'icu_mortality')
    assert not paired_cache_matches(saved,expected,{**manifest,'split_counts':{'test':{'n':100,'positive':1}}},'icu_mortality')
    assert not paired_cache_matches(saved.iloc[::-1],expected,manifest,'icu_mortality')
    wrong=saved.copy();wrong.loc[1,'y_true']=0
    assert not paired_cache_matches(wrong,expected,manifest,'icu_mortality')


def test_tabular_input_validation():
    from src.models.train_utils import create_and_fit
    x=np.zeros((4,2),dtype=np.float32);y=np.array([0,1,0,1])
    with pytest.raises(ValueError,match='不属于五种表格模型'):
        create_and_fit('unknown',{},x,y,x,y)
    with pytest.raises(ValueError,match='二维'):
        create_and_fit('lr',{},x[:,None,:],y,x,y)
