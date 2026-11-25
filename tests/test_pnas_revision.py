"""本轮时间边界、患者配对和序列权限的回归检查。"""
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.pnas_revision.repair_cutoff_treatment import output_rows
from scripts.pnas_revision.run_gru_seq import channels
from src.data.transforms import split_cohort
from src.models.gru_seq_model import GRUSequenceModel


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


def test_sequence_groups_and_future_event_boundary():
    values=np.full((3,6,2),np.nan,dtype=np.float32)
    counts=np.zeros((3,6,2),dtype=np.uint16)
    treatment=np.zeros((3,6,3),dtype=np.uint8)
    values[0,0,0]=10;counts[0,0,0]=1
    values[1,1,1]=20;counts[1,1,1]=1
    values[2,2,0]=30;counts[2,2,0]=1
    base,_=channels(values,counts,treatment,np.array([0,1,2]),3)
    future_values=values.copy();future_counts=counts.copy();future_t=treatment.copy()
    future_values[0,5,0]=900;future_counts[0,5,0]=1;future_t[0,5,0]=1
    after,_=channels(future_values,future_counts,future_t,np.array([0,1,2]),3)
    for group in base:np.testing.assert_array_equal(base[group],after[group])
    assert base['D'].shape==(3,3,2)
    assert base['O'].shape==(3,3,6)
    assert base['DT'].shape==(3,3,5)
    assert base['F'].shape==(3,3,11)
    np.testing.assert_array_equal(base['D'],base['DT'][:,:,:2])
    np.testing.assert_array_equal(base['D'],base['F'][:,:,:2])
    assert np.all(base['DT'][:,:,2:]==0)


def test_gru_rejects_single_step_and_2d():
    model=GRUSequenceModel(2,0)
    with pytest.raises(ValueError):model(torch.ones(2,2))
    with pytest.raises(ValueError):model(torch.ones(2,1,2))
    assert model(torch.ones(2,3,2),torch.zeros(2,0)).shape==(2,)


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


def test_old_single_step_gru_is_retired():
    from src.models.train_utils import create_and_fit
    x=np.zeros((4,2),dtype=np.float32);y=np.array([0,1,0,1])
    with pytest.raises(ValueError,match='真实序列'):
        create_and_fit('gru',{},x,y,x,y)


def test_smoke_and_formal_paths_and_wrong_sample_cache():
    from pathlib import Path
    from scripts.pnas_revision.run_tabular import cell_path,paired_cache_matches
    root=Path('/tmp/pnas_test')
    smoke=cell_path(root,'icu_mortality','lr',3,42,True)
    formal=cell_path(root,'icu_mortality','lr',3,42,False)
    assert smoke!=formal and '/smoke/' in str(smoke) and '/formal/' in str(formal)
    expected=pd.DataFrame({'subject_id':[1,2,3],'stay_id':[11,12,13],'icu_mortality':[0,1,0]})
    saved=pd.DataFrame({'pair_id':[1,2,3],'stay_id':[11,12,13],'y_true':[0,1,0]})
    manifest={'split_counts':{'test':{'n':3,'positive':1}}}
    assert paired_cache_matches(saved,expected,manifest,'icu_mortality')
    assert not paired_cache_matches(saved,expected,{**manifest,'split_counts':{'test':{'n':100,'positive':1}}},'icu_mortality')
    assert not paired_cache_matches(saved.iloc[::-1],expected,manifest,'icu_mortality')
    wrong=saved.copy();wrong.loc[1,'y_true']=0
    assert not paired_cache_matches(wrong,expected,manifest,'icu_mortality')
