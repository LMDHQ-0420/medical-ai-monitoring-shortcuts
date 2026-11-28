"""外部截点前录入诊断分层，仅作探索性分析。"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.feature_groups import PREDICTOR_SETS
from src.exp1.trainer import MODELS,TASKS,WINDOWS,SEEDS,atomic_csv
from src.exp5.external_validation import read_external
from src.formal.statistics import assess,gains

STRATA=("infection","cardiovascular","respiratory","renal","other_recorded","not_recorded")


def collect(root:Path,data_root:Path,protocol:dict):
    side=pd.read_parquet(data_root/"eicu_diagnosis_timing.parquet")
    rows=[]
    for task,model,window_h,seed in itertools.product(TASKS,MODELS,WINDOWS,SEEDS):
        frames={}
        for predictor_set in PREDICTOR_SETS:
            _,frame=read_external(root,task,model,window_h,seed,predictor_set)
            frames[predictor_set]=frame
        base=frames["full_model"]
        flags=side.loc[side.window_h.eq(window_h),["stay_id",*STRATA]]
        joined=base.merge(flags,on="stay_id",validate="one_to_one")
        if len(joined)!=len(base):
            raise ValueError("诊断侧表未覆盖外部预测")
        for stratum in STRATA:
            part=joined.loc[joined[stratum]]
            ix=part.index.to_numpy()
            y=part.y_true.to_numpy(dtype=int)
            metrics={name:assess(y,frame.loc[ix,"probability"].to_numpy()) for name,frame in frames.items()}
            rows.append({"analysis_id":protocol["analysis_id"],"task":task,"model":model,
                         "window_h":window_h,"seed":seed,"diagnosis_stratum":stratum,
                         "n":len(y),"n_positive":int(y.sum()),"n_negative":int(len(y)-y.sum()),
                         "evaluable":metrics["full_model"]["evaluable"],
                         "unevaluable_reason":metrics["full_model"]["unevaluable_reason"],
                         "stratum_definition":"diagnosis/admissionDx entered before cutoff; categories may overlap",
                         "external_endpoint_status":"primary" if task in ("icu_mortality","inhosp_mortality") else "exploratory",
                         **gains(metrics)})
    atomic_csv(root/"exp2/diagnosis_metrics.csv.gz",pd.DataFrame(rows))
    return len(rows)
