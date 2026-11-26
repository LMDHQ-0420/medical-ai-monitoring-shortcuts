"""从 eICU 原始偏移时间构建与 MIMIC 同 schema 的逐小时外部序列。"""
from __future__ import annotations
import json,os
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
from src.data.eicu_cohort import VITAL_COLS,LAB_NAMES
from src.data.treatment_events import eicu_hourly,sha256
from src.data.mimic_sequence import VARS
VID={x:i for i,x in enumerate(VARS)}
LAB={x:y for x,y in LAB_NAMES.items() if y in VID}

def scan(path,cols,kind,ids):
 parts=[]
 for n,chunk in enumerate(pd.read_csv(path,usecols=cols,chunksize=400_000,low_memory=False)):
  sub=chunk.loc[chunk.patientunitstayid.isin(ids)].copy()
  if kind=='lab':
   sub=sub[sub.labname.isin(LAB)]
   offset=pd.to_numeric(sub.labresultoffset,errors='coerce')
   revised=pd.to_numeric(sub.labresultrevisedoffset,errors='coerce')
   avail=np.maximum(offset,revised.fillna(offset))
   sub['var']=sub.labname.map(LAB)
   sub['value']=pd.to_numeric(sub.labresult,errors='coerce')
   sub['minute']=avail
   sub=sub[['patientunitstayid','minute','var','value']]
  else:
   sub['minute']=pd.to_numeric(sub.observationoffset,errors='coerce')
   sub=sub.melt(id_vars=['patientunitstayid','minute'],value_vars=list(VITAL_COLS),var_name='source_var',value_name='value')
   sub['var']=sub.source_var.map(VITAL_COLS)
   sub['value']=pd.to_numeric(sub.value,errors='coerce')
   sub=sub[['patientunitstayid','minute','var','value']]
  sub=sub[sub.minute.ge(0)&sub.minute.lt(1440)&sub.value.notna()].copy()
  if sub.empty:continue
  sub['hour']=np.floor(sub.minute/60).astype('int8')
  g=sub.groupby(['patientunitstayid','hour','var'],sort=False)
  last=g.tail(1)[['patientunitstayid','hour','var','minute','value']]
  count=g.size().rename('count').reset_index()
  parts.append(last.merge(count,on=['patientunitstayid','hour','var']))
  if (n+1)%30==0:print(path.name,'chunks',n+1,flush=True)
 if not parts:return pd.DataFrame(columns=['patientunitstayid','hour','var','minute','value','count'])
 allrows=pd.concat(parts,ignore_index=True)
 counts=allrows.groupby(['patientunitstayid','hour','var'])['count'].sum().rename('count').reset_index()
 last=allrows.sort_values('minute').groupby(['patientunitstayid','hour','var'],sort=False).tail(1).drop(columns='count')
 return last.merge(counts,on=['patientunitstayid','hour','var'],validate='one_to_one')

def build_sequence():
 raw=ROOT/'data/raw/eicu/eicu-collaborative-research-database-2.0';out=ROOT/'data/processed'
 tab=pd.read_parquet(out/'eicu_24h.parquet',columns=['stay_id','subject_id','age','gender','hospital_id'])
 ids=tab.stay_id.to_numpy();idset=set(ids)
 vital=scan(raw/'vitalPeriodic.csv.gz',['patientunitstayid','observationoffset',*VITAL_COLS],'vital',idset)
 print('eICU vital hourly rows',len(vital),flush=True)
 lab=scan(raw/'lab.csv.gz',['patientunitstayid','labresultoffset','labresultrevisedoffset','labname','labresult'],'lab',idset)
 print('eICU lab hourly rows',len(lab),flush=True)
 events=pd.concat([vital,lab],ignore_index=True);n=len(ids);h=24;v=len(VARS)
 values=np.full((n,h,v),np.nan,dtype=np.float32);counts=np.zeros((n,h,v),dtype=np.uint16)
 ix=pd.Index(ids).get_indexer(events.patientunitstayid);good=ix>=0;events=events.loc[good];ix=ix[good]
 var=events['var'].map(VID).to_numpy(dtype='int32');hour=events.hour.to_numpy(dtype='int32')
 values[ix,hour,var]=events.value.to_numpy(dtype=np.float32)
 counts[ix,hour,var]=np.minimum(events['count'].to_numpy(),65535).astype(np.uint16)
 temp=values[:,:,VID['temp']];f=(temp>=80)&(temp<=115);converted=int(f.sum());temp[f]=(temp[f]-32)*5/9
 bad=(temp<25)|(temp>45);invalid=int(bad.sum());temp[bad]=np.nan
 _,treat_hourly=eicu_hourly();treatment=np.zeros((n,h,3),dtype=np.uint8)
 for j,name in enumerate(['antibiotic','vaso','vent']):
  wide=treat_hourly[name].unstack('hour').reindex(index=ids,columns=range(24),fill_value=0).fillna(0)
  treatment[:,:,j]=(wide.to_numpy()>0).astype(np.uint8)
 path=out/'eicu_hourly.npz';tmp=out/'eicu_hourly.tmp.npz'
 np.savez_compressed(tmp,stay_id=ids,subject_id=tab.subject_id.to_numpy(),hospital_id=tab.hospital_id.to_numpy(),
                     age=tab.age.to_numpy(),values=values,counts=counts,treatment=treatment,
                     static_D=tab[['age','gender']].to_numpy(dtype=np.float32))
 os.replace(tmp,path)
 manifest={'dataset':'eicu','n_stays':n,'hours':24,'variables':VARS,'treatment_channels':['antibiotic','vaso','vent'],
  'measurement_available_time':'observationoffset for vitals; max(labresultoffset,labresultrevisedoffset) for labs',
  'treatment_time':'drugstartoffset','temperature_unit':'Celsius; 80–115 F converted; outside 25–45 C missing',
  'temperature_converted':converted,'temperature_invalid_set_missing':invalid,
  'coverage_fraction_of_stay_hours':{name:float((counts[:,:,j]>0).mean()) for j,name in enumerate(VARS)},
  'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}
 (out/'eicu_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print('eICU sequence complete',n,len(VARS),flush=True)
