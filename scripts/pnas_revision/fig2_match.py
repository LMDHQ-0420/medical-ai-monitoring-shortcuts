"""在冻结模型的测试集内，按截点前临床状态匹配高低观察强度患者。"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.transforms import split_cohort
from src.data.feature_groups import OBSERVATION_FEATURES
from src.exp2.oii import OIIScaler

VITALS = ("hr", "sbp", "dbp", "temp", "rr", "spo2")
CONT = ("age",) + tuple(v + "_value" for v in VITALS)
MASK = tuple(v + "_observed" for v in VITALS)
BALANCE = CONT + MASK + ("gender", "admission_source")


def smd(frame: pd.DataFrame, selected: np.ndarray | None = None) -> list[dict]:
    d = frame if selected is None else frame.loc[selected]
    high = d.loc[d.high_oii]
    low = d.loc[~d.high_oii]
    rows = []
    for col in BALANCE:
        a = pd.to_numeric(high[col], errors="coerce").dropna()
        b = pd.to_numeric(low[col], errors="coerce").dropna()
        va = float(a.var()) if len(a) > 1 else 0.0
        vb = float(b.var()) if len(b) > 1 else 0.0
        denom = np.sqrt((va + vb) / 2)
        delta = float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
        score = delta / denom if denom > 0 else (0.0 if delta == 0 else np.nan)
        rows.append({"covariate": col, "smd": score, "high_n_observed": len(a), "low_n_observed": len(b),
                     "high_missing": int(high[col].isna().sum()), "low_missing": int(low[col].isna().sum())})
    return rows


def pair(frame: pd.DataFrame, caliper_sd: float, age_limit: float, train_logit_sd: float,
         train_median: np.ndarray, train_scale: np.ndarray) -> pd.DataFrame:
    """严格按性别及入院来源分组，按 logit 倾向与临床状态距离无放回配对。"""
    frame = frame.reset_index(drop=True)
    high = frame.index[frame.high_oii].to_numpy()
    low = frame.index[~frame.high_oii].to_numpy()
    # 预先按训练集拟合的尺度计算距离，缺失指标作为独立维度。
    continuous = frame[list(CONT)].to_numpy(dtype=float)
    missing = frame[list(MASK)].to_numpy(dtype=float)
    state = np.column_stack(((np.where(np.isnan(continuous), train_median, continuous) - train_median) / train_scale, missing))
    gender = frame.gender.to_numpy(dtype=int)
    source = frame.admission_source.to_numpy(dtype=int)
    age = frame.age.to_numpy(dtype=float)
    stay = frame.stay_id.to_numpy(dtype=int)
    subject = frame.subject_id.to_numpy(dtype=int)
    oii = frame.oii.to_numpy(dtype=float)
    logit = frame.logit_ps.to_numpy(dtype=float)
    groups = {}
    for i in low:
        key = (gender[i], source[i])
        groups.setdefault(key, []).append((logit[i], int(i)))
    for values in groups.values():
        values.sort()
    high = sorted(high, key=lambda i: (logit[i], stay[i]))
    pairs = []
    for i in high:
        key = (gender[i], source[i])
        available = groups.get(key)
        if not available:
            continue
        target = logit[i]
        pos = bisect.bisect_left(available, (target, -1))
        candidate_positions = []
        left, right = pos - 1, pos
        while len(candidate_positions) < 40 and (left >= 0 or right < len(available)):
            if left >= 0 and (right >= len(available) or abs(available[left][0] - target) <= abs(available[right][0] - target)):
                j = left; left -= 1
            else:
                j = right; right += 1
            if abs(available[j][0] - target) > caliper_sd * train_logit_sd:
                break
            other = available[j][1]
            if abs(age[i] - age[other]) <= age_limit:
                candidate_positions.append(j)
        if not candidate_positions:
            continue
        best = min(candidate_positions, key=lambda k: (
            float(np.sum((state[i] - state[available[k][1]]) ** 2)),
            abs(available[k][0] - target), stay[available[k][1]],
        ))
        _, j = available.pop(best)
        pairs.append({"pair_index": len(pairs), "high_stay_id": stay[i],
                      "high_pair_id": subject[i], "low_stay_id": stay[j],
                      "low_pair_id": subject[j], "high_oii": oii[i], "low_oii": oii[j],
                      "high_logit_ps": target, "low_logit_ps": logit[j]})
    return pd.DataFrame(pairs)


def run(run_dir: Path) -> None:
    state = pd.read_parquet(ROOT / "data/processed/mimic_iv_fig2_clinical_state.parquet")
    diagnostics, all_pairs, thresholds = [], [], []
    for window in (3, 6, 12, 18, 24):
        cohort = pd.read_parquet(ROOT / f"data/processed/mimic_iv_{window}h.parquet")
        st = state.loc[state.window_h.eq(window)].drop(columns="window_h")
        cohort = cohort.merge(st, on="stay_id", how="left", validate="one_to_one")
        if cohort.prediction_cutoff.isna().any():
            raise ValueError(f"{window}h 临床状态未覆盖完整队列")
        for seed in (42, 43, 44, 45, 46):
            train, _, test = split_cohort(cohort, seed=seed)
            # 用训练集分位点限制显然不合理的生命体征极端值；缺失仍保留并另设指示。
            clipping = {}
            for vital in VITALS:
                column = vital + "_value"
                lower, upper = train[column].quantile([.01, .99]).to_numpy(dtype=float)
                clipping[column] = [float(lower), float(upper)]
                train[column] = train[column].clip(lower, upper)
                test[column] = test[column].clip(lower, upper)
            cols = list(OBSERVATION_FEATURES)
            scaler = OIIScaler().fit(train, cols)
            train_oii = scaler.transform(train, cols).clip(0, 1).mean(axis=1)
            threshold = float(train_oii.median())
            test = test.copy()
            test["oii"] = scaler.transform(test, cols).clip(0, 1).mean(axis=1).to_numpy()
            test["high_oii"] = test.oii.gt(threshold)
            train = train.copy()
            train["high_oii"] = train_oii.gt(threshold).to_numpy()
            columns = list(CONT + MASK + ("gender", "admission_source"))
            train_x = train[columns].copy()
            test_x = test[columns].copy()
            for x in (train_x, test_x):
                for col in MASK:
                    x[col] = x[col].astype(float)
            propensity = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       LogisticRegression(max_iter=1000, random_state=seed))
            propensity.fit(train_x, train.high_oii.astype(int))
            ps_train = propensity.predict_proba(train_x)[:, 1]
            ps_test = propensity.predict_proba(test_x)[:, 1]
            logit_train = np.log(np.clip(ps_train, 1e-6, 1 - 1e-6) / np.clip(1 - ps_train, 1e-6, 1))
            test["logit_ps"] = np.log(np.clip(ps_test, 1e-6, 1 - 1e-6) / np.clip(1 - ps_test, 1e-6, 1))
            sd = float(np.std(logit_train))
            train_cont = train[list(CONT)].to_numpy(dtype=float)
            train_median = np.nanmedian(train_cont, axis=0)
            train_median[~np.isfinite(train_median)] = 0
            train_scale = np.nanstd(train_cont, axis=0)
            train_scale[~np.isfinite(train_scale) | (train_scale < 1e-6)] = 1
            before = smd(test)
            candidates = []
            for caliper in (0.1, 0.2, 0.3):
                for age_limit in (5, 10):
                    p = pair(test, caliper, age_limit, sd, train_median, train_scale)
                    selected = test.stay_id.isin(pd.concat([p.high_stay_id, p.low_stay_id]) if len(p) else [])
                    after = smd(test, selected.to_numpy()) if len(p) else []
                    finite = [abs(r["smd"]) for r in after if np.isfinite(r["smd"])]
                    max_smd = max(finite) if finite else np.inf
                    candidates.append((max_smd <= 0.1 and len(p) >= 100, max_smd, len(p), caliper, age_limit, p, after))
            feasible = [x for x in candidates if x[2] >= 100]
            if not feasible:
                raise ValueError(f"{window}h seed{seed} 没有至少 100 对合格测试患者")
            passed = [x for x in feasible if x[0]]
            chosen = max(passed, key=lambda x: (x[2], -x[1])) if passed else min(feasible, key=lambda x: (x[1], -x[2]))
            _, max_smd, n_pairs, caliper, age_limit, p, after = chosen
            p["window_h"] = window; p["seed"] = seed
            all_pairs.append(p)
            for stage, records in (("before", before), ("after", after)):
                diagnostics.extend({"window_h": window, "seed": seed, "stage": stage, **r} for r in records)
            thresholds.append({"window_h": window, "seed": seed, "threshold_train_median": threshold,
                               "oii_features": cols, "propensity_fit": "train_only",
                               "matching_population": "test_only", "n_test": len(test),
                               "n_high": int(test.high_oii.sum()), "n_low": int((~test.high_oii).sum()),
                               "n_pairs": n_pairs, "n_matched": 2 * n_pairs,
                               "n_unmatched": len(test) - 2 * n_pairs,
                               "caliper_sd": caliper, "age_limit_years": age_limit,
                               "max_abs_smd_after": max_smd, "balance_pass": bool(max_smd <= 0.1),
                               "training_quantile_clip_1_99": clipping,
                               "selection_rule": "highest retained pairs among candidates with max |SMD|<=0.1; otherwise smallest max |SMD|; no outcomes or ODI used"})
            print("matched", window, seed, n_pairs, "max SMD", round(max_smd, 3), flush=True)
    out = run_dir / "fig2_matching"
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    if pairs.duplicated(["window_h", "seed", "high_stay_id"]).any() or pairs.duplicated(["window_h", "seed", "low_stay_id"]).any():
        raise ValueError("重复匹配患者")
    pairs.to_csv(out / "matched_pairs.csv.gz", index=False, compression="gzip")
    pd.DataFrame(diagnostics).to_csv(out / "balance_diagnostics.csv", index=False)
    pd.DataFrame(thresholds).to_json(out / "matching_protocol.json", orient="records", indent=2)
    print("total pairs", len(pairs), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    run(ap.parse_args().run_dir.resolve())
