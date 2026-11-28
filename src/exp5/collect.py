"""从冻结外部概率重算医院效应、反转和概率误差；不拟合模型。"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.feature_groups import PREDICTOR_SETS
from src.exp1.trainer import MODELS, TASKS, WINDOWS, SEEDS, atomic_csv
from src.exp5.external_validation import read_external
from src.formal.statistics import assess, gains, seed_synchronous_probability_interval

BASE_N, BASE_POS, BASE_NEG = 100, 5, 5
SENS_POS, SENS_NEG = 50, 50


def collect(root: Path, protocol: dict):
    internal = pd.read_csv(root / "exp1/gain_metrics.csv.gz")
    rows, eligibility = [], []
    for task, model, window_h, seed in itertools.product(TASKS, MODELS, WINDOWS, SEEDS):
        frames, manifests = {}, {}
        for predictor_set in PREDICTOR_SETS:
            manifest, frame = read_external(root, task, model, window_h, seed, predictor_set)
            frames[predictor_set], manifests[predictor_set] = frame, manifest
        reference = frames["clinical_state"]
        for name, frame in frames.items():
            if not np.array_equal(frame.subject_id.astype(str), reference.subject_id.astype(str)) or not np.array_equal(frame.stay_id, reference.stay_id) or not np.array_equal(frame.y_true, reference.y_true):
                raise ValueError(f"外部输入组未对齐: {task}/{model}/{window_h}/{seed}/{name}")
        internal_row = internal.loc[internal.task.eq(task) & internal.model.eq(model) &
                                    internal.window_h.eq(window_h) & internal.seed.eq(seed)]
        if len(internal_row) != 1:
            raise ValueError("缺少对应内部增益")
        internal_row = internal_row.iloc[0]
        for hospital_id, group in reference.groupby("hospital_id", dropna=False):
            ix = group.index.to_numpy()
            y = group.y_true.to_numpy(dtype=int)
            metrics = {name: assess(y, frame.loc[ix, "probability"].to_numpy())
                       for name, frame in frames.items()}
            contrast = gains(metrics)
            n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
            eligible = len(y) >= BASE_N and n_pos >= BASE_POS and n_neg >= BASE_NEG
            sensitivity_eligible = n_pos >= SENS_POS and n_neg >= SENS_NEG
            key = {"analysis_id": protocol["analysis_id"], "task": task,
                   "model": model, "window_h": window_h, "seed": seed,
                   "hospital_id": hospital_id, "state_representation": "first_measurements",
                   "cohort_id": manifests["full_model"]["cohort_id"],
                   "endpoint_version": protocol["endpoint_version"],
                   "evaluation_id": manifests["full_model"]["evaluation_id"],
                   "n": len(y), "n_positive": n_pos, "n_negative": n_neg,
                   "evaluable": bool(metrics["full_model"]["evaluable"]),
                   "unevaluable_reason": metrics["full_model"]["unevaluable_reason"],
                   "eligible": eligible, "sensitivity_eligible": sensitivity_eligible}
            rows.append({**key,
                         **{f"auroc_{name}": met["auroc"] for name, met in metrics.items()},
                         **{f"brier_{name}": met["brier"] for name, met in metrics.items()},
                         **{f"fit_id_{name}": man["fit_id"] for name, man in manifests.items()},
                         **contrast,
                         "internal_observation_gain": internal_row.observation_gain,
                         "internal_treatment_gain": internal_row.treatment_gain,
                         "internal_combined_gain": internal_row.combined_gain,
                         "observation_transfer_change": contrast["observation_gain"] - internal_row.observation_gain,
                         "treatment_transfer_change": contrast["treatment_gain"] - internal_row.treatment_gain,
                         "combined_transfer_change": contrast["combined_gain"] - internal_row.combined_gain})
            if model == MODELS[0] and seed == SEEDS[0]:
                eligibility.append({"task": task, "window_h": window_h, "hospital_id": hospital_id,
                                    "n": len(y), "n_positive": n_pos, "n_negative": n_neg,
                                    "eligible": eligible, "sensitivity_eligible": sensitivity_eligible,
                                    "base_rule": f"n>={BASE_N}, positive>={BASE_POS}, negative>={BASE_NEG}",
                                    "sensitivity_rule": f"positive>={SENS_POS}, negative>={SENS_NEG}"})
    out = root / "exp5"
    hospital = pd.DataFrame(rows)
    atomic_csv(out / "hospital_metrics.csv.gz", hospital)
    pd.DataFrame(eligibility).to_csv(out / "hospital_eligibility.csv", index=False)
    reverse = []
    eligible_rows = hospital.loc[hospital.eligible].copy()
    for (task, hospital_id), section in eligible_rows.groupby(["task", "hospital_id"], dropna=False):
        config = section.groupby(["model", "window_h"], as_index=False).agg(
            internal_mean=("internal_observation_gain", "mean"),
            external_mean=("observation_gain", "mean"), n_seeds=("seed", "nunique"))
        config = config.loc[config.n_seeds.eq(len(SEEDS)) & config.internal_mean.gt(0)]
        denominator = len(config)
        numerator = int(config.external_mean.lt(0).sum())
        reverse.append({"analysis_id": protocol["analysis_id"], "task": task,
                        "hospital_id": hospital_id, "n_configs": denominator,
                        "n_reversed": numerator,
                        "reversal_fraction": numerator / denominator if denominator else np.nan,
                        "unevaluable_reason": "" if denominator else "zero_positive_internal_config"})
    atomic_csv(out / "hospital_reversal.csv.gz", pd.DataFrame(reverse))
    probability = hospital.loc[hospital.task.eq("inhosp_mortality") & hospital.window_h.eq(24) &
                               hospital.model.isin(("lr", "lgbm"))].copy()
    metric_rows = []
    for predictor_set in ("clinical_state_treatment", "full_model"):
        cols = ["analysis_id", "task", "model", "window_h", "seed", "hospital_id", "n",
                "n_positive", "n_negative", "evaluable", "unevaluable_reason", "eligible",
                "sensitivity_eligible", "cohort_id", "endpoint_version", "evaluation_id"]
        part = probability[cols].copy()
        part["predictor_set"] = predictor_set
        part["state_representation"] = "first_measurements"
        part["fit_id"] = probability[f"fit_id_{predictor_set}"].to_numpy()
        part["auroc"] = probability[f"auroc_{predictor_set}"].to_numpy()
        part["brier"] = probability[f"brier_{predictor_set}"].to_numpy()
        metric_rows.append(part)
    atomic_csv(out / "probability_metrics.csv.gz", pd.concat(metric_rows, ignore_index=True))
    cols = ["analysis_id", "task", "model", "window_h", "seed", "hospital_id",
            "n", "n_positive", "n_negative", "eligible", "sensitivity_eligible",
            "evaluable", "unevaluable_reason", "observation_gain", "brier_improvement",
            "fit_id_clinical_state_treatment", "fit_id_full_model", "evaluation_id"]
    contrasts=probability[cols].copy()
    interval_rows=[]
    for model in ('lr','lgbm'):
        seed_frames={}
        for seed in SEEDS:
            _, a=read_external(root,'inhosp_mortality',model,24,seed,'clinical_state_treatment')
            _, b=read_external(root,'inhosp_mortality',model,24,seed,'full_model')
            if not np.array_equal(a.stay_id,b.stay_id) or not np.array_equal(a.y_true,b.y_true):
                raise ValueError('概率分析两组患者未对齐')
            seed_frames[seed]=(a,b)
        reference=seed_frames[SEEDS[0]][0]
        for seed in SEEDS[1:]:
            if not np.array_equal(reference.stay_id,seed_frames[seed][0].stay_id):
                raise ValueError('概率分析不同种子患者未对齐')
        for hospital_id, part in reference.groupby('hospital_id',dropna=False):
            section=probability.loc[probability.model.eq(model)&probability.hospital_id.eq(hospital_id)]
            if len(section)!=len(SEEDS) or not section.eligible.all():
                continue
            ix=part.index.to_numpy()
            y=part.y_true.to_numpy(dtype=int)
            a=np.stack([seed_frames[seed][0].loc[ix,'probability'].to_numpy() for seed in SEEDS])
            b=np.stack([seed_frames[seed][1].loc[ix,'probability'].to_numpy() for seed in SEEDS])
            interval=seed_synchronous_probability_interval(
                y,a,b,protocol['n_bootstrap'],f'probability/{model}/{hospital_id}')
            interval_rows.append({'model':model,'hospital_id':hospital_id,
                                  'seed_mean_observation_gain':float(section.observation_gain.mean()),
                                  'seed_mean_brier_improvement':float(section.brier_improvement.mean()),
                                  **interval})
    intervals=pd.DataFrame(interval_rows)
    if len(intervals):
        contrasts=contrasts.merge(intervals,on=['model','hospital_id'],how='left',validate='many_to_one')
    atomic_csv(out / "probability_contrasts.csv.gz", contrasts)
    return len(hospital), len(reverse), len(probability)
