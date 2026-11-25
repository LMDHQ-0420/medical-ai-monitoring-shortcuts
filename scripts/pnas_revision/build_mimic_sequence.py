"""从 MIMIC 原始时间戳构建逐小时真实测量序列；不拼接窗口汇总表。"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.mimic_cohort import VITAL_ITEMIDS,LAB_ITEMIDS
from scripts.pnas_revision.repair_cutoff_treatment import mimic_hourly
VARS=list(VITAL_ITEMIDS)+[x for x in LAB_ITEMIDS if x in ('creatinine','lactate','wbc','hgb','sodium','potassium','bilirubin','inr')]
VID={name:i for i,name in enumerate(VARS)}
MIMIC_VITAL={item:name for name,items in VITAL_ITEMIDS.items() for item in items}
MIMIC_LAB={item:name for name,items in LAB_ITEMIDS.items() if name in VID for item in items}

def scan(path,columns,id_col,lookup,item_map,chunk_size=400_000):
    parts=[]
    for n,chunk in enumerate(pd.read_csv(path,usecols=columns,chunksize=chunk_size,low_memory=False)):
        sub=chunk.loc[chunk.itemid.isin(item_map) & chunk[id_col].isin(lookup.index)].copy()
        if sub.empty:continue
        sub=sub.join(lookup,on=id_col,how='inner')
        chart=pd.to_datetime(sub.charttime,errors='coerce',format='mixed')
        store=pd.to_datetime(sub.storetime,errors='coerce',format='mixed')
        avail=chart.where(chart>=store,store).fillna(chart).fillna(store)
        hour=np.floor((avail-sub.intime).dt.total_seconds()/3600)
        val=pd.to_numeric(sub.valuenum,errors='coerce')
        good=hour.notna()&(hour>=0)&(hour<24)&val.notna()
        sub=sub.loc[good,['stay_id','itemid']].copy()
        if sub.empty:continue
        sub['hour']=hour.loc[good].astype('int8');sub['time']=avail.loc[good];sub['value']=val.loc[good].astype('float32')
        sub['var']=sub.itemid.map(item_map)
        # MIMIC 223761 为华氏温度，转为摄氏度。
        fahrenheit=sub.itemid.eq(223761)
        sub.loc[fahrenheit,'value']=(sub.loc[fahrenheit,'value']-32)*5/9
        sub=sub.sort_values('time')
        g=sub.groupby(['stay_id','hour','var'],sort=False)
        last=g.tail(1)[['stay_id','hour','var','time','value']]
        counts=g.size().rename('count').reset_index()
        parts.append(last.merge(counts,on=['stay_id','hour','var']))
        if (n+1)%30==0:print(path.name,'chunks',n+1,flush=True)
    if not parts:return pd.DataFrame(columns=['stay_id','hour','var','time','value','count'])
    allrows=pd.concat(parts,ignore_index=True)
    totals=allrows.groupby(['stay_id','hour','var'])['count'].sum().rename('count').reset_index()
    last=allrows.sort_values('time').groupby(['stay_id','hour','var'],sort=False).tail(1).drop(columns='count')
    return last.merge(totals,on=['stay_id','hour','var'],validate='one_to_one')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
    out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    tab=pd.read_parquet(ROOT/'data/processed/mimic_iv_24h.parquet',columns=['stay_id','hadm_id','subject_id','age','gender'])
    stay=pd.read_csv(ROOT/'data/raw/mimic_iv/mimic-iv-2.2/icu/icustays.csv.gz',usecols=['stay_id','intime']);stay.intime=pd.to_datetime(stay.intime)
    tab=tab.merge(stay,on='stay_id',validate='one_to_one');ids=tab.stay_id.to_numpy()
    bystay=tab.set_index('stay_id')[['intime']];byhadm=tab.set_index('hadm_id')[['stay_id','intime']]
    raw=ROOT/'data/raw/mimic_iv/mimic-iv-2.2'
    vital=scan(raw/'icu/chartevents.csv.gz',['stay_id','charttime','storetime','itemid','valuenum'],'stay_id',bystay,MIMIC_VITAL)
    print('MIMIC vital hourly rows',len(vital),flush=True)
    lab=scan(raw/'hosp/labevents.csv.gz',['hadm_id','charttime','storetime','itemid','valuenum'],'hadm_id',byhadm,MIMIC_LAB)
    print('MIMIC lab hourly rows',len(lab),flush=True)
    events=pd.concat([vital,lab],ignore_index=True)
    n=len(ids);h=24;v=len(VARS)
    values=np.full((n,h,v),np.nan,dtype=np.float32);counts=np.zeros((n,h,v),dtype=np.uint16)
    index=pd.Index(ids).get_indexer(events.stay_id)
    good=index>=0
    events=events.loc[good];index=index[good]
    var=events['var'].map(VID).to_numpy(dtype='int32');hour=events.hour.to_numpy(dtype='int32')
    values[index,hour,var]=events.value.to_numpy(dtype=np.float32)
    counts[index,hour,var]=np.minimum(events['count'].to_numpy(),65535).astype(np.uint16)
    # 统一温度为摄氏度：混合源中仍可能有华氏读数；不合理数值作缺失。
    temp=values[:,:,VID['temp']]
    fahrenheit=(temp>=80)&(temp<=115)
    temp[fahrenheit]=(temp[fahrenheit]-32)*5/9
    temp[(temp<25)|(temp>45)]=np.nan
    treatment_ids,treat_hourly=mimic_hourly()
    treatment=np.zeros((n,h,3),dtype=np.uint8)
    for j,name in enumerate(['antibiotic','vaso','vent']):
        if name not in treat_hourly:continue
        wide=treat_hourly[name].unstack('hour').reindex(index=ids,columns=range(24),fill_value=0).fillna(0)
        treatment[:,:,j]=(wide.to_numpy()>0).astype(np.uint8)
    temp_path=out/'mimic_iv_hourly.tmp.npz'
    np.savez_compressed(temp_path,stay_id=ids,subject_id=tab.subject_id.to_numpy(),
                        values=values,counts=counts,treatment=treatment,
                        static_D=tab[['age','gender']].to_numpy(dtype=np.float32))
    os.replace(temp_path,out/'mimic_iv_hourly.npz')
    coverage={name:float((counts[:,:,j]>0).mean()) for j,name in enumerate(VARS)}
    manifest={'dataset':'mimic_iv','time_grid':'hourly','n_stays':n,'hours':24,'variables':VARS,
              'treatment_channels':['antibiotic','vaso','vent'],'measurement_available_time':'max(charttime,storetime)',
              'treatment_time':'starttime (order entry time unavailable for prescriptions)',
              'temperature_unit':'Celsius; item 223761 and remaining 80–115 F values converted; outside 25–45 C treated as missing',
              'coverage_fraction_of_stay_hours':coverage,'source':'raw CSV.gz with patient cohort filter',
              'path':str((out/'mimic_iv_hourly.npz').relative_to(ROOT))}
    (out/'mimic_iv_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('sequence complete',n,len(VARS),flush=True)
if __name__=='__main__':main()
