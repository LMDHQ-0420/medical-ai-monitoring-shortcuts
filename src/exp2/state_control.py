"""24 小时住院死亡状态增强：四个冻结预测同步比较。"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.exp1.collect import read_group
from src.exp1.trainer import atomic_csv, fit_group
from src.formal.statistics import assess

VITALS = ("hr", "sbp", "dbp", "temp", "rr", "spo2")
MODELS = ("lr", "lgbm")
SEEDS = (42, 43, 44, 45, 46)
SETS = ("clinical_state_treatment", "full_model")


def prepare_side_table(legacy_side: Path, new_24h: Path, destination: Path) -> pd.DataFrame:
    cohort = pd.read_parquet(new_24h, columns=["stay_id"])
    side = pd.read_parquet(legacy_side)
    side = side.loc[side.window_h.eq(24)].merge(cohort, on="stay_id", validate="one_to_one")
    if len(side) != len(cohort) or side.duplicated(["stay_id", "window_h"]).any():
        raise ValueError("状态侧表与新 24h 风险集不一致")
    rows = []
    for vital in VITALS:
        seen = side[f"{vital}_observed"].fillna(False).astype(bool)
        available = pd.to_datetime(side.loc[seen, f"{vital}_available_time"])
        chart = pd.to_datetime(side.loc[seen, f"{vital}_charttime"])
        cutoff = pd.to_datetime(side.loc[seen, "prediction_cutoff"])
        if (available >= cutoff).any() or (chart > available).any():
            raise ValueError(f"{vital} 最新值违反截点可用性")
        rows.append({"vital": vital, "window_h": 24, "n": len(side),
                     "n_observed": int(seen.sum()), "coverage": float(seen.mean())})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    side.to_parquet(temp, index=False)
    temp.replace(destination)
    return pd.DataFrame(rows)


def train_enhanced(config: dict, data_root: Path, output_root: Path,
                   protocol: dict, registry: pd.DataFrame) -> None:
    for model, seed, predictor_set in itertools.product(MODELS, SEEDS, SETS):
        manifest = fit_group(config, data_root, output_root, protocol,
                             "inhosp_mortality", model, 24, seed,
                             "first_plus_latest", predictor_set, registry)
        print("state_enhanced", model, seed, predictor_set,
              manifest["fit_source"], flush=True)


def collect(output_root: Path, protocol: dict) -> tuple[int, int]:
    metric_rows, contrast_rows = [], []
    count = protocol["n_bootstrap"]
    for model, seed in itertools.product(MODELS, SEEDS):
        probabilities = {}
        reference = None
        manifests = {}
        for state, predictor_set in itertools.product(("first_measurements", "first_plus_latest"), SETS):
            manifest, frame = read_group(output_root, "inhosp_mortality", model, 24,
                                         seed, state, predictor_set)
            if reference is None:
                reference = frame
            elif not np.array_equal(frame.subject_id, reference.subject_id) or not np.array_equal(frame.y_true, reference.y_true):
                raise ValueError("状态增强四组患者与标签不一致")
            key = (state, predictor_set)
            probabilities[key] = frame.probability.to_numpy()
            manifests[key] = manifest
            metric_rows.append({"analysis_id": protocol["analysis_id"], "task": "inhosp_mortality",
                                "model": model, "window_h": 24, "seed": seed,
                                "state_representation": state, "predictor_set": predictor_set,
                                "cohort_id": manifest["cohort_id"], "split_id": manifest["split_id"],
                                "endpoint_version": manifest["endpoint_version"],
                                "fit_id": manifest["fit_id"], "evaluation_id": manifest["evaluation_id"],
                                **assess(frame.y_true.to_numpy(), frame.probability.to_numpy())})
        y = reference.y_true.to_numpy(dtype=int)
        def values(ix):
            result = {key: assess(y[ix], p[ix])["auroc"] for key, p in probabilities.items()}
            base = result[("first_measurements", "full_model")] - result[("first_measurements", "clinical_state_treatment")]
            enhanced = result[("first_plus_latest", "full_model")] - result[("first_plus_latest", "clinical_state_treatment")]
            return {"baseline_reference_auroc_change": result[("first_plus_latest", "clinical_state_treatment")] - result[("first_measurements", "clinical_state_treatment")],
                    "baseline_full_auroc_change": result[("first_plus_latest", "full_model")] - result[("first_measurements", "full_model")],
                    "observation_gain_before": base,
                    "observation_gain_after": enhanced,
                    "observation_gain_change": enhanced - base}
        observed = values(np.arange(len(y)))
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"state/{model}/{seed}".encode()).digest()[:8], "big"))
        draws = {key: [] for key in observed}
        for _ in range(count):
            ix = rng.integers(0, len(y), len(y))
            if len(np.unique(y[ix])) < 2:
                continue
            for key, value in values(ix).items():
                draws[key].append(value)
        manifest = manifests[("first_plus_latest", "full_model")]
        contrast_rows.append({"analysis_id": protocol["analysis_id"], "task": "inhosp_mortality",
                              "model": model, "window_h": 24, "seed": seed,
                              "cohort_id": manifest["cohort_id"], "split_id": manifest["split_id"],
                              "endpoint_version": manifest["endpoint_version"],
                              "evaluation_id": manifest["evaluation_id"],
                              "n": len(y), "n_positive": int(y.sum()), "n_negative": int(len(y) - y.sum()),
                              "bootstrap_unit": "patient", "bootstrap_requested": count,
                              "bootstrap_valid": len(draws["observation_gain_change"]), **observed,
                              **{f"{key}_ci_low": float(np.quantile(sample, .025)) for key, sample in draws.items()},
                              **{f"{key}_ci_high": float(np.quantile(sample, .975)) for key, sample in draws.items()}})
    out = output_root / "exp2"
    atomic_csv(out / "state_control_metrics.csv.gz", pd.DataFrame(metric_rows))
    atomic_csv(out / "state_control_contrasts.csv.gz", pd.DataFrame(contrast_rows))
    return len(metric_rows), len(contrast_rows)
