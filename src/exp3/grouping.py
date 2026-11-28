"""内部来源分解及同配置外部变化；只读取 Exp1 与 Exp5。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.exp1.trainer import atomic_csv

GAIN_NAMES = ("observation", "treatment", "combined")


def collect(root: Path, protocol: dict) -> tuple[int, int, int]:
    internal = pd.read_csv(root / "exp1/gain_metrics.csv.gz")
    external = pd.read_csv(root / "exp5/hospital_metrics.csv.gz")
    cols = ["task", "model", "window_h", "seed"]
    if internal.duplicated(cols).any():
        raise ValueError("内部配置重复")
    source = internal[[*cols, "analysis_id", "n", "n_positive", "n_negative",
                       "observation_gain", "treatment_gain", "combined_gain",
                       "brier_improvement", "evaluation_id",
                       "fit_id_clinical_state", "fit_id_clinical_state_treatment", "fit_id_full_model"]].copy()
    out = root / "exp3"
    atomic_csv(out / "gain_components.csv.gz", source)
    merged = external.merge(source[cols + [f"{g}_gain" for g in GAIN_NAMES]],
                            on=cols, suffixes=("_external", "_internal"),
                            validate="many_to_one")
    if len(merged) != len(external):
        raise ValueError("外部医院配置缺少内部来源")
    components = merged[[*cols, "hospital_id", "n", "n_positive", "n_negative",
                         "eligible", "sensitivity_eligible", "evaluable", "unevaluable_reason",
                         "evaluation_id", "analysis_id_external"]].rename(columns={"analysis_id_external": "analysis_id"}).copy()
    for gain in GAIN_NAMES:
        components[f"{gain}_gain_internal"] = merged[f"{gain}_gain_internal"]
        components[f"{gain}_gain_external"] = merged[f"{gain}_gain_external"]
        components[f"{gain}_external_minus_internal"] = (components[f"{gain}_gain_external"] -
                                                           components[f"{gain}_gain_internal"])
    if not np.allclose(components["combined_external_minus_internal"],
                       components["observation_external_minus_internal"] +
                       components["treatment_external_minus_internal"], equal_nan=True, atol=1e-12):
        raise ValueError("转移增益加和失败")
    atomic_csv(out / "transfer_components.csv.gz", components)
    eligible = components.loc[components.eligible].copy()
    # 医院×算法×窗口先跨种子平均；缺种子不视为完整配置。
    group_cols = ["task", "hospital_id", "model", "window_h"]
    agg = {f"{g}_external_minus_internal": "mean" for g in GAIN_NAMES}
    config = eligible.groupby(group_cols, dropna=False).agg(
        **{name: (name, "mean") for name in agg}, n_seeds=("seed", "nunique")).reset_index()
    config = config.loc[config.n_seeds.eq(5)]
    hospital = config.groupby(["task", "hospital_id"], dropna=False).agg(
        **{name: (name, "mean") for name in agg}, n_configs=("model", "size")).reset_index()
    summary = hospital.groupby("task").agg(
        **{name: (name, "mean") for name in agg},
        n_hospitals=("hospital_id", "nunique"), n_configs=("n_configs", "sum")).reset_index()
    summary.insert(0, "analysis_id", protocol["analysis_id"])
    if not np.allclose(summary["combined_external_minus_internal"],
                       summary["observation_external_minus_internal"] +
                       summary["treatment_external_minus_internal"], equal_nan=True, atol=1e-12):
        raise ValueError("医院等权汇总加和失败")
    atomic_csv(out / "transfer_summary.csv.gz", summary)
    return len(source), len(components), len(summary)
