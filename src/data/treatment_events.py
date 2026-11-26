"""扫描原始治疗事件，按预测截点重算 T 和处方数，覆盖现有处理数据。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
from src.data.mimic_cohort import VASOPRESSOR_ITEMIDS,MV_ITEMIDS,ANTIBIOTIC_KEYWORDS
from src.data.eicu_cohort import VASOPRESSOR_KEYWORDS
WINDOWS=(3,6,12,18,24)
FEATURES=('orders_count_24h','antibiotics_24h','vasopressors_24h','mechanical_ventilation_24h')

def read_events(path,columns,map_df,id_name,time_name,drug_col=None,code_col=None,kind='order'):
    parts=[]
    for chunk in pd.read_csv(path,usecols=columns,chunksize=500_000,low_memory=False):
        sub=chunk.loc[chunk[id_name].isin(map_df.index)].copy()
        if sub.empty: continue
        sub=sub.join(map_df,on=id_name,how='inner')
        t=pd.to_datetime(sub[time_name],errors='coerce',format='mixed')
        hour=np.floor((t-sub.intime).dt.total_seconds()/3600)
        good=hour.notna() & (hour>=0) & (hour<24)
        sub=sub.loc[good].copy();sub['hour']=hour.loc[good].astype('int8')
        if sub.empty:continue
        if kind=='order':
            sub['count']=1
            sub['antibiotic']=sub[drug_col].astype(str).str.lower().str.contains('|'.join(ANTIBIOTIC_KEYWORDS),na=False).astype('int8')
            use=['count','antibiotic']
        elif kind=='vaso':
            sub['vaso']=sub[code_col].isin(VASOPRESSOR_ITEMIDS).astype('int8');use=['vaso']
        else:
            sub['vent']=sub[code_col].isin(MV_ITEMIDS).astype('int8');use=['vent']
        parts.append(sub.groupby(['stay_id','hour'],sort=False)[use].sum())
    if not parts:return pd.DataFrame()
    return pd.concat(parts).groupby(level=[0,1]).sum()

def mimic_hourly():
    ids=pd.read_parquet(ROOT/'data/processed/mimic_iv_24h.parquet',columns=['stay_id','hadm_id'])
    stay=pd.read_csv(ROOT/'data/raw/mimic_iv/mimic-iv-2.2/icu/icustays.csv.gz',usecols=['stay_id','intime'])
    stay.intime=pd.to_datetime(stay.intime)
    ids=ids.merge(stay,on='stay_id',validate='one_to_one')
    raw=ROOT/'data/raw/mimic_iv/mimic-iv-2.2'
    by_hadm=ids.set_index('hadm_id')[['stay_id','intime']]
    by_stay=ids.set_index('stay_id')[['intime']]
    pr=read_events(raw/'hosp/prescriptions.csv.gz',['hadm_id','starttime','drug'],by_hadm,'hadm_id','starttime',drug_col='drug',kind='order')
    print('MIMIC prescriptions aggregated',flush=True)
    ie=read_events(raw/'icu/inputevents.csv.gz',['stay_id','starttime','itemid'],by_stay,'stay_id','starttime',code_col='itemid',kind='vaso')
    print('MIMIC inputevents aggregated',flush=True)
    pe=read_events(raw/'icu/procedureevents.csv.gz',['stay_id','starttime','itemid'],by_stay,'stay_id','starttime',code_col='itemid',kind='vent')
    print('MIMIC procedureevents aggregated',flush=True)
    allh=pr.join(ie,how='outer').join(pe,how='outer').fillna(0)
    return ids.stay_id.to_numpy(),allh

def eicu_hourly():
    ids=pd.read_parquet(ROOT/'data/processed/eicu_24h.parquet',columns=['stay_id']).stay_id.to_numpy()
    path=ROOT/'data/raw/eicu/eicu-collaborative-research-database-2.0/medication.csv.gz'
    parts=[]
    for chunk in pd.read_csv(path,usecols=['patientunitstayid','drugstartoffset','drugname'],chunksize=500_000,low_memory=False):
        sub=chunk.loc[chunk.patientunitstayid.isin(ids)].copy()
        offset=pd.to_numeric(sub.drugstartoffset,errors='coerce')
        good=offset.notna() & (offset>=0)&(offset<1440)
        sub=sub.loc[good].copy();sub['hour']=np.floor(offset.loc[good]/60).astype('int8')
        if sub.empty:continue
        name=sub.drugname.astype(str).str.lower()
        sub['count']=1
        sub['antibiotic']=name.str.contains('|'.join(ANTIBIOTIC_KEYWORDS),na=False).astype('int8')
        sub['vaso']=name.str.contains('|'.join(VASOPRESSOR_KEYWORDS),na=False).astype('int8')
        sub['vent']=name.str.contains('ventilat',na=False).astype('int8')
        parts.append(sub.groupby(['patientunitstayid','hour'],sort=False)[['count','antibiotic','vaso','vent']].sum())
    allh=pd.concat(parts).groupby(level=[0,1]).sum()
    allh.index=allh.index.set_names(['stay_id','hour'])
    print('eICU medication aggregated',flush=True)
    return ids,allh

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def output_rows(ids,hourly,window):
    result=pd.DataFrame({'stay_id':ids})
    for source,target in [('count','orders_count_24h'),('antibiotic','antibiotics_24h'),('vaso','vasopressors_24h'),('vent','mechanical_ventilation_24h')]:
        if source not in hourly:hourly[source]=0
        wide=hourly[source].unstack('hour').reindex(index=ids,columns=range(24),fill_value=0).fillna(0)
        value=wide.cumsum(axis=1)[window-1].to_numpy()
        result[target]=(value if source=='count' else value>0).astype('int32')
    return result
