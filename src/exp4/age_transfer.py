"""四年龄组内部、外部汇集和逐医院配对增益。"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.feature_groups import PREDICTOR_SETS
from src.exp1.collect import read_group
from src.exp1.trainer import MODELS,TASKS,WINDOWS,SEEDS,atomic_csv
from src.exp5.external_validation import read_external
from src.formal.statistics import assess,gains

AGE_GROUPS = ("<45", "45-64", "65-80", ">80")


def age_label(values):
    return pd.cut(values, bins=[-np.inf,45,65,81,np.inf], labels=AGE_GROUPS, right=False)


def collect(root: Path, protocol: dict):
    metrics, contrasts, eligibility = [], [], []
    for task,model,window_h,seed in itertools.product(TASKS,MODELS,WINDOWS,SEEDS):
        for dataset, reader in (("mimic_iv",read_group),("eicu",read_external)):
            frames, manifests = {}, {}
            for predictor_set in PREDICTOR_SETS:
                manifest, frame = reader(root,task,model,window_h,seed,predictor_set) if dataset == "eicu" else reader(root,task,model,window_h,seed,"first_measurements",predictor_set)
                frames[predictor_set],manifests[predictor_set]=frame,manifest
            reference=frames["full_model"].copy()
            for name,frame in frames.items():
                if not np.array_equal(frame.stay_id,reference.stay_id) or not np.array_equal(frame.y_true,reference.y_true):
                    raise ValueError("年龄分析输入组不成对")
            reference["age_group"]=age_label(reference.age)
            sets=(("internal",np.nan,reference),) if dataset=="mimic_iv" else (("external_pooled",np.nan,reference),*[("external_hospital",hospital,part) for hospital,part in reference.groupby("hospital_id",dropna=False)])
            for environment,hospital_id,part in sets:
                strata={}
                for age_group,subset in part.groupby("age_group",observed=True):
                    ix=subset.index.to_numpy()
                    y=subset.y_true.to_numpy(dtype=int)
                    n_pos,n_neg=int(y.sum()),int(len(y)-y.sum())
                    status={"n":len(y),"n_positive":n_pos,"n_negative":n_neg,
                            "eligible":len(y)>=20 and n_pos>=5 and n_neg>=5,
                            "evaluable":n_pos>0 and n_neg>0}
                    eligibility.append({"task":task,"model":model,"window_h":window_h,"seed":seed,
                                        "environment":environment,"hospital_id":hospital_id,
                                        "age_group":str(age_group),**status})
                    met={name:assess(y,frame.loc[ix,"probability"].to_numpy()) for name,frame in frames.items()}
                    g=gains(met)
                    row={"analysis_id":protocol["analysis_id"],"task":task,"model":model,
                         "window_h":window_h,"seed":seed,"environment":environment,
                         "hospital_id":hospital_id,"age_group":str(age_group),
                         "endpoint_version":protocol["endpoint_version"],
                         "evaluation_id":manifests["full_model"]["evaluation_id"],
                         **status,**{f"auroc_{name}":value["auroc"] for name,value in met.items()},**g}
                    metrics.append(row)
                    strata[str(age_group)]=row
                if all(group in strata and strata[group]["eligible"] for group in ("<45",">80")):
                    contrasts.append({"analysis_id":protocol["analysis_id"],"task":task,"model":model,
                                      "window_h":window_h,"seed":seed,"environment":environment,
                                      "hospital_id":hospital_id,
                                      "age_gap_observation_gain":strata[">80"]["observation_gain"]-strata["<45"]["observation_gain"],
                                      "n_young":strata["<45"]["n"],"n_old":strata[">80"]["n"],
                                      "positive_young":strata["<45"]["n_positive"],
                                      "positive_old":strata[">80"]["n_positive"]})
    out=root/"exp4"
    metric_table=pd.DataFrame(metrics)
    contrast_table=pd.DataFrame(contrasts)
    atomic_csv(out/"age_metrics.csv.gz",metric_table)
    atomic_csv(out/"age_contrasts.csv.gz",contrast_table)
    pd.DataFrame(eligibility).to_csv(out/"age_eligibility.csv",index=False)
    internal=contrast_table.loc[contrast_table.environment.eq("internal"),["task","model","window_h","seed","age_gap_observation_gain"]].rename(columns={"age_gap_observation_gain":"internal_age_gap"})
    external=contrast_table.loc[contrast_table.environment.ne("internal")].merge(internal,on=["task","model","window_h","seed"],validate="many_to_one")
    external["age_gap_transfer_change"]=external.age_gap_observation_gain-external.internal_age_gap
    atomic_csv(out/"age_transfer_contrasts.csv.gz",external)
    return len(metric_table),len(contrast_table),len(external)
