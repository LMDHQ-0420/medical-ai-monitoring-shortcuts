"""跨种子(42-46)汇总 exp1-exp5 结果，复算 idea_report.md 中引用的全部统计量。

用法：python3 scripts/aggregate_multiseed.py [--seeds 42 43 44 45 46]
输出：results/multiseed_summary.json（供人工核对/写作时引用）+ stdout 汇总表。
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TASKS = ["icu_mortality", "inhosp_mortality", "aki", "sepsis_deterioration", "escalation"]
MODELS = ["lr", "rf", "xgb", "lgbm", "mlp"]
WINDOWS = [3, 6, 12, 18, 24]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 实验1 (results/exp1): ODI = auroc(mixed) - auroc(disease)，另附 observation 独立预测力
# ---------------------------------------------------------------------------
def aggregate_exp1(seeds):
    rows = []
    for task, model, win in product(TASKS, MODELS, WINDOWS):
        for seed in seeds:
            base = RESULTS_DIR / "exp1" / task
            d = load_json(base / "disease" / model / f"seed{seed}" / f"{win}h_metrics.json")
            m = load_json(base / "mixed" / model / f"seed{seed}" / f"{win}h_metrics.json")
            o = load_json(base / "observation" / model / f"seed{seed}" / f"{win}h_metrics.json")
            if d is None or m is None:
                continue
            odi = m["auroc"] - d["auroc"]
            obs_beats_disease = (o is not None) and (o["auroc"] > d["auroc"])
            rows.append(dict(task=task, model=model, window=win, seed=seed, odi=odi,
                              obs_beats_disease=obs_beats_disease))
    df = pd.DataFrame(rows)
    return df


def summarize_exp1(df):
    # 每个 task x model x window 先跨 seed 取均值，再统计支持率（与报告口径一致：150组网格，每组是单 seed42 的一个值；
    # 多种子版本这里先跨 seed 求均值 ODI，再统计 150 组中均值为正的比例，同时报告跨 seed 的方差信息）
    grp = df.groupby(["task", "model", "window"])["odi"].agg(["mean", "std", "count"]).reset_index()
    n_total = len(grp)
    n_pos = int((grp["mean"] > 0).sum())
    by_task = grp.groupby("task")["mean"].mean().sort_values()
    by_model = grp.groupby("model")["mean"].mean().sort_values()
    obs_grp = df.groupby(["task", "model", "window"])["obs_beats_disease"].mean().reset_index()
    obs_rate_by_task = obs_grp.groupby("task")["obs_beats_disease"].mean().sort_values()
    return dict(
        n_total=n_total, n_pos=n_pos, pct_pos=n_pos / n_total * 100,
        by_task_mean_odi=by_task.to_dict(),
        by_model_mean_odi=by_model.to_dict(),
        obs_beats_disease_rate_by_task=obs_rate_by_task.to_dict(),
        per_combo_seed_std_mean=float(grp["std"].mean(skipna=True)),
        exceptions=grp[grp["mean"] <= 0][["task", "model", "window", "mean"]].to_dict("records"),
    )


# ---------------------------------------------------------------------------
# 实验2 (results/exp2): PSM-ODI (severity control ablation)
# ---------------------------------------------------------------------------
def aggregate_exp2(seeds):
    rows = []
    for task, model, win in product(TASKS, MODELS, WINDOWS):
        for seed in seeds:
            p = (RESULTS_DIR / "exp2" / task / "mixed" / model / f"seed{seed}"
                 / f"{win}h_severity_control.json")
            data = load_json(p)
            if data is None:
                continue
            psm = data.get("psm", {})
            strat = data.get("stratification", {})
            rows.append(dict(
                task=task, model=model, window=win, seed=seed,
                psm_odi=psm.get("odi"), psm_ci_lower=psm.get("ci_lower"),
                strat_all_pos=all(v > 0 for v in strat.values()) if strat else None,
            ))
    return pd.DataFrame(rows)


def summarize_exp2(df):
    if df.empty:
        return dict(n_total=0)
    grp = df.groupby(["task", "model", "window"]).agg(
        psm_odi=("psm_odi", "mean"),
        psm_ci_lower=("psm_ci_lower", "mean"),
        strat_all_pos=("strat_all_pos", "mean"),
    ).reset_index()
    n_total = len(grp)
    n_pos = int((grp["psm_odi"] > 0).sum())
    n_sig_pos = int((grp["psm_ci_lower"] > 0).sum())
    n_strat_all_pos = int((grp["strat_all_pos"] >= 0.5).sum())
    by_task = grp.groupby("task").apply(
        lambda g: pd.Series({"sig_rate": (g["psm_ci_lower"] > 0).mean() * 100, "mean_psm_odi": g["psm_odi"].mean()})
    ).sort_values("sig_rate")
    by_window = grp.groupby("window").apply(
        lambda g: pd.Series({"sig_rate": (g["psm_ci_lower"] > 0).mean() * 100})
    ).sort_values("sig_rate")
    by_model = grp.groupby("model").apply(
        lambda g: pd.Series({"sig_rate": (g["psm_ci_lower"] > 0).mean() * 100})
    ).sort_values("sig_rate")
    return dict(
        n_total=n_total, n_pos=n_pos, pct_pos=n_pos / n_total * 100,
        n_sig_pos=n_sig_pos, pct_sig_pos=n_sig_pos / n_total * 100,
        n_strat_all_pos=n_strat_all_pos, pct_strat_all_pos=n_strat_all_pos / n_total * 100,
        by_task=by_task.to_dict("index"),
        by_window_sig_rate=by_window["sig_rate"].to_dict(),
        by_model_sig_rate=by_model["sig_rate"].to_dict(),
    )


# ---------------------------------------------------------------------------
# 实验3 (results/exp3): 变体A/B ODI (variable grouping sensitivity)
# ---------------------------------------------------------------------------
def aggregate_exp3(seeds):
    rows = []
    for task, model, win in product(TASKS, MODELS, WINDOWS):
        for seed in seeds:
            p = (RESULTS_DIR / "exp3" / task / "mixed" / model / f"seed{seed}"
                 / f"{win}h_grouping_ablation.json")
            data = load_json(p)
            if data is None:
                continue
            va, vb = data.get("variant_A", {}), data.get("variant_B", {})
            rows.append(dict(
                task=task, model=model, window=win, seed=seed,
                odi_A=va.get("odi"), ci_lower_A=va.get("ci_lower"),
                odi_B=vb.get("odi"), ci_lower_B=vb.get("ci_lower"),
            ))
    return pd.DataFrame(rows)


def summarize_exp3(df):
    if df.empty:
        return dict(n_total=0)
    grp = df.groupby(["task", "model", "window"]).agg(
        odi_A=("odi_A", "mean"), ci_lower_A=("ci_lower_A", "mean"),
        odi_B=("odi_B", "mean"), ci_lower_B=("ci_lower_B", "mean"),
    ).reset_index()
    n_total = len(grp)
    n_sig_A = int((grp["ci_lower_A"] > 0).sum())
    n_sig_B = int((grp["ci_lower_B"] > 0).sum())
    n_all_pos = int(((grp["odi_A"] > 0) & (grp["odi_B"] > 0)).sum())
    by_task = grp.groupby("task").apply(
        lambda g: pd.Series({
            "sig_rate_A": (g["ci_lower_A"] > 0).mean() * 100,
            "mean_odi_B": g["odi_B"].mean(), "mean_odi_A": g["odi_A"].mean(),
        })
    ).sort_values("sig_rate_A")
    by_window = grp.groupby("window").apply(
        lambda g: pd.Series({"sig_rate_A": (g["ci_lower_A"] > 0).mean() * 100})
    ).sort_values("sig_rate_A")
    return dict(
        n_total=n_total,
        n_sig_B=n_sig_B, pct_sig_B=n_sig_B / n_total * 100,
        n_sig_A=n_sig_A, pct_sig_A=n_sig_A / n_total * 100,
        n_all_pos=n_all_pos, pct_all_pos=n_all_pos / n_total * 100,
        by_task=by_task.to_dict("index"),
        by_window_sig_rate_A=by_window["sig_rate_A"].to_dict(),
    )


# ---------------------------------------------------------------------------
# 实验4 (results/exp4): mimic_gap = ODI(>80) - ODI(<45)，来自 summary json
# ---------------------------------------------------------------------------
def aggregate_exp4(seeds):
    rows = []
    for task, model, win in product(TASKS, MODELS, WINDOWS):
        for seed in seeds:
            p = (RESULTS_DIR / "exp4" / "summary" / model / f"seed{seed}"
                 / f"{win}h_odi_age_transfer_summary.json")
            data = load_json(p)
            if data is None or task not in data:
                continue
            v = data[task]
            gap = v.get("mimic_gap_old_minus_young", v.get("mimic_gap"))
            if gap is None:
                continue
            rows.append(dict(task=task, model=model, window=win, seed=seed, mimic_gap=gap))
    return pd.DataFrame(rows)


def summarize_exp4(df):
    if df.empty:
        return dict(n_total=0)
    grp = df.groupby(["task", "model", "window"])["mimic_gap"].agg(["mean", "std"]).reset_index()
    n_total = len(grp)
    n_pos = int((grp["mean"] > 0).sum())
    by_task = grp.groupby("task").apply(
        lambda g: pd.Series({"support_rate": (g["mean"] > 0).mean() * 100, "mean_gap": g["mean"].mean()})
    ).sort_values("support_rate")
    by_window = grp.groupby("window").apply(
        lambda g: pd.Series({"support_rate": (g["mean"] > 0).mean() * 100})
    ).sort_values("support_rate")
    by_model = grp.groupby("model").apply(
        lambda g: pd.Series({"support_rate": (g["mean"] > 0).mean() * 100})
    ).sort_values("support_rate")
    return dict(
        n_total=n_total, n_pos=n_pos, pct_pos=n_pos / n_total * 100,
        by_task=by_task.to_dict("index"),
        by_window_support_rate=by_window["support_rate"].to_dict(),
        by_model_support_rate=by_model["support_rate"].to_dict(),
    )


# ---------------------------------------------------------------------------
# 实验5 (results/exp5): odi_shrinkage = odi_mimic - odi_eicu，
# 按 model x window 池化全部 task x hospital 行做单样本 t 检验
# ---------------------------------------------------------------------------
def aggregate_exp5(seeds):
    rows = []
    for model, win in product(MODELS, WINDOWS):
        for seed in seeds:
            for task in TASKS:
                p = (RESULTS_DIR / "exp5" / "pooled" / "mixed" / model / f"seed{seed}"
                     / f"{win}h_by_hospital.csv")
                if not p.exists():
                    continue
                d = pd.read_csv(p)
                d = d[d["task"] == task]
                if d.empty:
                    continue
                d = d.assign(model=model, window=win, seed=seed)
                rows.append(d[["model", "window", "seed", "task", "odi_shrinkage"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_exp5(df):
    if df.empty:
        return dict(n_total=0)
    results = []
    for (model, win), sub in df.groupby(["model", "window"]):
        vals = sub["odi_shrinkage"].dropna().values
        if len(vals) < 2:
            continue
        mean = float(np.mean(vals))
        _, p = ttest_1samp(vals, 0.0)
        results.append(dict(model=model, window=win, mean=mean, p=float(p), n=len(vals)))
    r = pd.DataFrame(results)
    n_total = len(r)
    n_pos = int((r["mean"] > 0).sum())
    n_sig_pos = int(((r["mean"] > 0) & (r["p"] < 0.05)).sum())
    n_sig_neg = int(((r["mean"] < 0) & (r["p"] < 0.05)).sum())
    sig_neg = r[(r["mean"] < 0) & (r["p"] < 0.05)][["model", "window", "mean", "p"]].to_dict("records")
    by_model = r.groupby("model")["mean"].mean().sort_values()
    by_window = r.groupby("window")["mean"].mean().sort_values()
    return dict(
        n_total=n_total, n_pos=n_pos, pct_pos=n_pos / n_total * 100 if n_total else 0,
        n_sig_pos=n_sig_pos, pct_sig_pos=n_sig_pos / n_total * 100 if n_total else 0,
        n_sig_neg=n_sig_neg, sig_neg_combos=sig_neg,
        by_model_mean=by_model.to_dict(), by_window_mean=by_window.to_dict(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    seeds = args.seeds

    print(f"聚合种子: {seeds}\n")

    out = {}

    df1 = aggregate_exp1(seeds)
    out["exp1"] = summarize_exp1(df1)
    print("\n=== 实验1 (results/exp1, RQ1, ODI=mixed-disease) ===")
    print(json.dumps(out["exp1"], indent=2, ensure_ascii=False, default=str))

    df2 = aggregate_exp2(seeds)
    out["exp2"] = summarize_exp2(df2)
    print("\n=== 实验2 (results/exp2, 严重度控制消融, PSM-ODI) ===")
    print(json.dumps(out["exp2"], indent=2, ensure_ascii=False, default=str))

    df3 = aggregate_exp3(seeds)
    out["exp3"] = summarize_exp3(df3)
    print("\n=== 实验3 (results/exp3, 变量分组敏感性, 变体A/B) ===")
    print(json.dumps(out["exp3"], indent=2, ensure_ascii=False, default=str))

    df4 = aggregate_exp4(seeds)
    out["exp4"] = summarize_exp4(df4)
    print("\n=== 实验4 (results/exp4, RQ3, mimic_gap) ===")
    print(json.dumps(out["exp4"], indent=2, ensure_ascii=False, default=str))

    df5 = aggregate_exp5(seeds)
    out["exp5"] = summarize_exp5(df5)
    print("\n=== 实验5 (results/exp5, RQ2, odi_shrinkage) ===")
    print(json.dumps(out["exp5"], indent=2, ensure_ascii=False, default=str))

    out_path = RESULTS_DIR / "multiseed_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写入汇总文件: {out_path}")


if __name__ == "__main__":
    main()
