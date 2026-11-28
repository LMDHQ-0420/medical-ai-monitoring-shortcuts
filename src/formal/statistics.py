"""同患者概率的可复算指标与同步重采样。"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def assess(y: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("标签或概率不完整")
    n_pos = int(y.sum())
    out = {"n": len(y), "n_positive": n_pos, "n_negative": len(y) - n_pos,
           "brier": float(brier_score_loss(y, p)) if len(y) else np.nan}
    if len(y) == 0 or n_pos == 0 or n_pos == len(y):
        return {**out, "auroc": np.nan, "evaluable": False,
                "unevaluable_reason": "empty" if not len(y) else "single_class"}
    return {**out, "auroc": float(roc_auc_score(y, p)), "evaluable": True,
            "unevaluable_reason": ""}


def gains(metrics: dict[str, dict]) -> dict:
    a = metrics["clinical_state"]["auroc"]
    b = metrics["clinical_state_treatment"]["auroc"]
    c = metrics["full_model"]["auroc"]
    result = {"observation_gain": c - b, "treatment_gain": b - a,
              "combined_gain": c - a,
              "brier_improvement": metrics["clinical_state_treatment"]["brier"] -
                                   metrics["full_model"]["brier"]}
    if np.isfinite(result["combined_gain"]) and not np.isclose(
            result["combined_gain"], result["observation_gain"] + result["treatment_gain"], atol=1e-12):
        raise ValueError("增益分解恒等式失败")
    return result


def paired_interval(y: np.ndarray, probabilities: dict[str, np.ndarray],
                    n_bootstrap: int, identity: str, pair_id: np.ndarray | None = None) -> dict:
    """同一抽样索引同时作用于所有模型；匹配分析可按 pair_id 成对抽样。"""
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    unique_ids = np.arange(len(y)) if pair_id is None else np.unique(pair_id)
    indices = None if pair_id is None else {key: np.flatnonzero(pair_id == key) for key in unique_ids}
    values = {"observation_gain": [], "treatment_gain": [], "combined_gain": [],
              "brier_improvement": []}
    for _ in range(n_bootstrap):
        draw = rng.choice(unique_ids, len(unique_ids), replace=True)
        ix = draw if indices is None else np.concatenate([indices[key] for key in draw])
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            continue
        current = {name: assess(yy, np.asarray(p)[ix]) for name, p in probabilities.items()}
        for name, value in gains(current).items():
            values[name].append(value)
    result = {"bootstrap_unit": "matched_pair" if pair_id is not None else "patient",
              "bootstrap_requested": n_bootstrap,
              "bootstrap_valid": len(values["observation_gain"])}
    for name, sample in values.items():
        result[f"{name}_ci_low"] = float(np.quantile(sample, .025)) if sample else np.nan
        result[f"{name}_ci_high"] = float(np.quantile(sample, .975)) if sample else np.nan
    return result


def seed_synchronous_probability_interval(y: np.ndarray, reference: np.ndarray,
                                          full: np.ndarray, n_bootstrap: int,
                                          identity: str) -> dict:
    """按患者同步重采样；五个种子分别算指标后再平均差值。"""
    y = np.asarray(y, dtype=int)
    reference = np.asarray(reference, dtype=float)
    full = np.asarray(full, dtype=float)
    if reference.shape != full.shape or reference.ndim != 2 or reference.shape[1] != len(y):
        raise ValueError('种子概率与患者维度不一致')
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(seed)
    values = {'observation_gain': [], 'brier_improvement': []}
    for _ in range(n_bootstrap):
        ix = rng.integers(0, len(y), len(y))
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            continue
        gains_by_seed, brier_by_seed = [], []
        for j in range(reference.shape[0]):
            a = assess(yy, reference[j, ix])
            b = assess(yy, full[j, ix])
            gains_by_seed.append(b['auroc'] - a['auroc'])
            brier_by_seed.append(a['brier'] - b['brier'])
        values['observation_gain'].append(float(np.mean(gains_by_seed)))
        values['brier_improvement'].append(float(np.mean(brier_by_seed)))
    result = {'bootstrap_unit': 'patient_shared_across_seeds_and_models',
              'bootstrap_requested': n_bootstrap,
              'bootstrap_valid': len(values['observation_gain'])}
    for key, sample in values.items():
        result[f'{key}_ci_low'] = float(np.quantile(sample, .025)) if sample else np.nan
        result[f'{key}_ci_high'] = float(np.quantile(sample, .975)) if sample else np.nan
    return result
