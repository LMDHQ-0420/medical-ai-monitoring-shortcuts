"""冻结模型在匹配对和入院情境中的配对评价。"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.feature_groups import PREDICTOR_SETS
from src.exp1.collect import read_group
from src.exp1.trainer import MODELS,TASKS,WINDOWS,SEEDS,atomic_csv
from src.formal.statistics import assess,gains,paired_interval


def collect(root: Path, data_root: Path, protocol: dict):
    pairs=pd.read_csv(root/"exp2/matched_pairs.csv.gz")
    metric_rows,contrast_rows,context_rows=[],[],[]
    for task,model,window_h,seed in itertools.product(TASKS,MODELS,WINDOWS,SEEDS):
        frames={}
        for predictor_set in PREDICTOR_SETS:
            _,frame=read_group(root,task,model,window_h,seed,"first_measurements",predictor_set)
            frames[predictor_set]=frame
        base=frames["full_model"]
        for frame in frames.values():
            if not np.array_equal(frame.stay_id,base.stay_id) or not np.array_equal(frame.y_true,base.y_true):
                raise ValueError("匹配评价输入组未对齐")
        selected=pairs.loc[pairs.window_h.eq(window_h)&pairs.seed.eq(seed)]
        high=selected[["pair_index","high_stay_id"]].rename(columns={"high_stay_id":"stay_id"});high["arm"]="high"
        low=selected[["pair_index","low_stay_id"]].rename(columns={"low_stay_id":"stay_id"});low["arm"]="low"
        chosen=pd.concat([high,low],ignore_index=True)
        joined=chosen.merge(base[["stay_id","subject_id","y_true"]],on="stay_id",how="left",validate="one_to_one")
        complete=joined.groupby("pair_index").y_true.count()
        kept=set(complete[complete.eq(2)].index)
        joined=joined.loc[joined.pair_index.isin(kept)].copy()
        if joined.empty:
            raise ValueError("任务可评估匹配对为空")
        if joined.duplicated(["pair_index","arm"]).any():
            raise ValueError("匹配对重复")
        y=joined.y_true.to_numpy(dtype=int)
        metrics,probabilities={},{}
        for predictor_set,frame in frames.items():
            aligned=joined[["stay_id"]].merge(frame[["stay_id","probability"]],on="stay_id",validate="one_to_one")
            probability=aligned.probability.to_numpy()
            probabilities[predictor_set]=probability
            met=assess(y,probability);metrics[predictor_set]=met
            metric_rows.append({"analysis_id":protocol["analysis_id"],"task":task,"model":model,
                                "window_h":window_h,"seed":seed,"predictor_set":predictor_set,
                                "state_representation":"first_measurements","n_pairs":len(kept),
                                "n_unmatched":len(selected)-len(kept),"matching_source":"train-only OII and state propensity; test-only pairing",
                                **met})
        ci=paired_interval(y,probabilities,protocol["n_bootstrap"],
                           f"matched/{task}/{model}/{window_h}/{seed}",joined.pair_index.to_numpy())
        contrast_rows.append({"analysis_id":protocol["analysis_id"],"task":task,"model":model,
                              "window_h":window_h,"seed":seed,"n_pairs":len(kept),
                              "n":len(joined),"n_positive":int(y.sum()),"n_negative":int(len(y)-y.sum()),
                              **gains(metrics),**ci})
        cohort=pd.read_parquet(data_root/f"mimic_iv_{window_h}h.parquet",columns=["stay_id","admission_source"])
        context=base.merge(cohort,on="stay_id",validate="one_to_one")
        for source,part in context.groupby("admission_source"):
            ix=part.index.to_numpy()
            yy=part.y_true.to_numpy(dtype=int)
            mets={name:assess(yy,frame.loc[ix,"probability"].to_numpy()) for name,frame in frames.items()}
            context_rows.append({"analysis_id":protocol["analysis_id"],"task":task,"model":model,
                                 "window_h":window_h,"seed":seed,"admission_source":source,
                                 "n":len(yy),"n_positive":int(yy.sum()),"n_negative":int(len(yy)-yy.sum()),
                                 "evaluable":mets["full_model"]["evaluable"],
                                 "unevaluable_reason":mets["full_model"]["unevaluable_reason"],
                                 **gains(mets)})
    out=root/"exp2"
    atomic_csv(out/"matched_metrics.csv.gz",pd.DataFrame(metric_rows))
    atomic_csv(out/"matched_contrasts.csv.gz",pd.DataFrame(contrast_rows))
    atomic_csv(out/"context_metrics.csv.gz",pd.DataFrame(context_rows))
    return len(metric_rows),len(contrast_rows),len(context_rows)
