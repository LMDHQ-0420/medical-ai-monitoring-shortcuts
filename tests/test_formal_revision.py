"""新正式协议中的模型身份、固定划分和同步重采样验收。"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.formal.identity import fit_identity, prediction_identity
from src.formal.splits import make_registry, split_frame
from src.formal.statistics import paired_interval, seed_synchronous_probability_interval


def _frame():
    return pd.DataFrame({'subject_id':range(100),'stay_id':range(1000,1100),
                         'label':[0,1]*50,'feature':np.arange(100,dtype=float),
                         'audit':0})


def test_fit_identity_tracks_only_actual_inputs():
    train=_frame()
    validation=train.iloc[:20].copy()
    args=('label',['feature'],'lr',42,{'C':1},'pre-v1','impl-v1')
    original=fit_identity(train,validation,*args,False)
    changed=train.copy();changed['audit']=1
    assert fit_identity(changed,validation,*args,False)==original
    changed.loc[0,'feature']+=1
    assert fit_identity(changed,validation,*args,False)!=original
    changed_val=validation.copy();changed_val.loc[0,'label']=1-changed_val.loc[0,'label']
    assert fit_identity(train,changed_val,*args,False)==original
    assert fit_identity(train,changed_val,*args,True)!=fit_identity(train,validation,*args,True)
    assert prediction_identity(original,train,['feature'])!=prediction_identity(original,changed,['feature'])


def test_split_registry_preserves_old_and_assigns_new(tmp_path:Path):
    old=_frame().iloc[:80]
    new=_frame()
    old.to_parquet(tmp_path/'old.parquet',index=False)
    new.to_parquet(tmp_path/'new.parquet',index=False)
    registry=make_registry(tmp_path/'old.parquet',tmp_path/'new.parquet',tmp_path/'registry.parquet')
    assert len(registry)==500
    assert registry.loc[registry.subject_id.lt(80),'source'].eq('legacy').all()
    assert registry.loc[registry.subject_id.ge(80),'source'].eq('new_hash').all()
    parts=split_frame(new,registry,42)
    assert sum(map(len,parts))==100
    assert not(set(parts[0].subject_id)&set(parts[1].subject_id))


def test_synchronous_patient_and_pair_bootstrap():
    y=np.array([0,1,0,1,0,1,0,1])
    p=np.array([.1,.9,.2,.8,.3,.7,.4,.6])
    same={'clinical_state':p,'clinical_state_treatment':p,'full_model':p}
    patient=paired_interval(y,same,50,'patient')
    pair=paired_interval(y,same,50,'pair',np.repeat(np.arange(4),2))
    assert patient['bootstrap_unit']=='patient'
    assert pair['bootstrap_unit']=='matched_pair'
    for result in (patient,pair):
        assert result['bootstrap_valid']>0
        assert result['observation_gain_ci_low']==0
        assert result['combined_gain_ci_high']==0
    multi=seed_synchronous_probability_interval(y,np.stack([p,p]),np.stack([p,p]),50,'seed')
    assert multi['observation_gain_ci_low']==0
    assert multi['brier_improvement_ci_high']==0
