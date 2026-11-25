"""核查既有队列、原始文件、划分与历史指标；不修改输入。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TASKS = ("icu_mortality", "inhosp_mortality", "aki", "sepsis_deterioration", "escalation")
MODELS = ("lr", "rf", "xgb", "lgbm", "gru", "mlp")
WINDOWS = (3, 6, 12, 18, 24)
SEEDS = (42, 43, 44, 45, 46)


def digest(values) -> str:
    """局部审计指纹，不在证据包中附原始标识。"""
    return hashlib.sha256("\n".join(map(str, sorted(values))).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inventory():
    rows = []
    paths = sorted((ROOT / "data/raw").rglob("*.csv.gz"))
    paths += sorted((ROOT / "data/processed").glob("*_?h.parquet"))
    paths += sorted((ROOT / "data/processed").glob("*_[12]?h.parquet"))
    paths += sorted((ROOT / "data/processed").glob("*_hourly.npz"))
    for path in sorted(set(paths)):
        rel = path.relative_to(ROOT).as_posix()
        dataset = "mimic_iv" if "mimic" in rel else "eicu"
        row_count = None
        columns = None
        full_sha = None
        if path.suffix == ".parquet":
            meta = pq.ParquetFile(path).metadata
            row_count = meta.num_rows
            columns = meta.num_columns
        elif path.suffix == ".npz":
            manifest = json.loads((path.parent / f"{dataset}_manifest.json").read_text())
            row_count = manifest["n_stays"]
            columns = len(manifest["variables"])
        if path.suffix in (".parquet", ".npz"):
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(2**20), b""):
                    h.update(block)
            full_sha = h.hexdigest()
        rows.append({"relative_path": rel, "dataset": dataset,
                     "format": "parquet" if path.suffix == ".parquet" else "npz" if path.suffix == ".npz" else "csv.gz",
                     "size_bytes": path.stat().st_size, "rows_metadata": row_count,
                     "columns_metadata": columns,
                     "version": "2.2" if dataset == "mimic_iv" else "2.0",
                     "purpose": "analysis cohort" if path.suffix == ".parquet" else "hourly sequence" if path.suffix == ".npz" else "source event/metadata",
                     "fingerprint_scope": "full SHA-256" if full_sha else "path+size+mtime; no full file hash",
                     "sha256": full_sha,
                     "mtime_ns": path.stat().st_mtime_ns})
    return rows


def audit_cohorts():
    cohort_rows, split_rows, age_rows, hospital_rows = [], [], [], []
    for dataset in ("mimic_iv", "eicu"):
        for window in WINDOWS:
            path = ROOT / f"data/processed/{dataset}_{window}h.parquet"
            if not path.exists():
                continue
            cols = ["subject_id", "stay_id", "age", *TASKS]
            if dataset == "mimic_iv":
                cols.append("hadm_id")
            else:
                cols.append("hospital_id")
            df = pd.read_parquet(path, columns=cols)
            base = {"dataset": dataset, "window_h": window,
                    "n_rows": len(df), "n_patients": df.subject_id.nunique(),
                    "n_stays": df.stay_id.nunique(),
                    "n_admissions": df.hadm_id.nunique() if "hadm_id" in df else None,
                    "patient_set_sha256": digest(df.subject_id.unique()),
                    "stay_set_sha256": digest(df.stay_id.unique())}
            for task in TASKS:
                valid = df.loc[df[task].notna()]
                cohort_rows.append({**base, "task": task, "n_valid": len(valid),
                                    "n_positive": int(valid[task].sum()),
                                    "n_negative": int((valid[task] == 0).sum())})
                for seed in SEEDS:
                    # 与当前已排序的 split_cohort 实现一致；同时记录指纹。
                    from src.data.transforms import split_cohort
                    tr, va, te = split_cohort(valid, seed=seed)
                    sets = [set(x.subject_id) for x in (tr, va, te)]
                    disjoint = not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
                    for split, sub in zip(("train", "validation", "test"), (tr, va, te)):
                        split_rows.append({"dataset": dataset, "task": task, "window_h": window,
                                           "seed": seed, "split": split, "n_rows": len(sub),
                                           "n_patients": sub.subject_id.nunique(),
                                           "n_positive": int(sub[task].sum()),
                                           "n_negative": int((sub[task] == 0).sum()),
                                           "patient_set_sha256": digest(sub.subject_id.unique()),
                                           "patient_order_sha256": hashlib.sha256("\n".join(map(str, sub.subject_id)).encode()).hexdigest(),
                                           "patient_disjoint": disjoint})
                if dataset == "mimic_iv":
                    bins = pd.cut(valid.age, bins=[0, 45, 65, 80, 120], labels=["<=45", "(45,65]", "(65,80]", ">80"])
                    for group in bins.cat.categories:
                        sub = valid.loc[bins == group]
                        age_rows.append({"task": task, "window_h": window,
                                         "age_group": group, "n": len(sub),
                                         "positive": int(sub[task].sum())})
                else:
                    g = valid.groupby("hospital_id").agg(n=(task, "size"), positive=(task, "sum"))
                    hospital_rows.append({"task": task, "window_h": window,
                                          "hospitals_total": len(g),
                                          "hospitals_n_ge_20": int((g.n >= 20).sum()),
                                          "hospitals_eligible_auc": int(((g.n >= 20) & (g.positive > 0) & (g.positive < g.n)).sum()),
                                          "eligible_patient_rows": int(g.loc[(g.n >= 20) & (g.positive > 0) & (g.positive < g.n), "n"].sum())})
    return cohort_rows, split_rows, age_rows, hospital_rows


def audit_old_results():
    rows = []
    for task in TASKS:
        for model in MODELS:
            for seed in SEEDS:
                for window in WINDOWS:
                    paths = {g: ROOT / f"results/exp1/{task}/{g}/{model}/seed{seed}/{window}h_metrics.json"
                             for g in ("disease", "observation", "mixed")}
                    data = {g: json.loads(p.read_text()) if p.exists() else None for g, p in paths.items()}
                    exp3 = ROOT / f"results/exp3/{task}/mixed/{model}/seed{seed}/{window}h_grouping_ablation.json"
                    e3 = json.loads(exp3.read_text()) if exp3.exists() else None
                    d, f = data["disease"], data["mixed"]
                    if not d or not f:
                        continue
                    original = e3.get("original", {}) if e3 else {}
                    odi = f["auroc"] - d["auroc"]
                    row = {"task": task, "model": model, "seed": seed, "window_h": window,
                           "n_D": d.get("n"), "n_O": data["observation"].get("n") if data["observation"] else None,
                           "n_F": f.get("n"), "auc_D": d["auroc"], "auc_F": f["auroc"],
                           "exp1_odi": odi, "exp3_original_odi": original.get("odi"),
                           "odi_difference": (odi - original["odi"]) if "odi" in original else None,
                           "metric_D_path": paths["disease"].relative_to(ROOT).as_posix(),
                           "metric_F_path": paths["mixed"].relative_to(ROOT).as_posix(),
                           "exp3_path": exp3.relative_to(ROOT).as_posix() if e3 else None}
                    if task == "icu_mortality" and model == "lr" and seed == 42 and window == 24:
                        for g in ("disease", "observation", "mixed"):
                            p = paths[g].with_name(f"{window}h_predictions.csv")
                            if p.exists():
                                pred = pd.read_csv(p)
                                row[f"prediction_{g}_n"] = len(pred)
                                row[f"prediction_{g}_positive"] = int(pred.y_true.sum())
                                row[f"prediction_{g}_auc"] = roc_auc_score(pred.y_true, pred.y_pred)
                    rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "input_inventory.csv", inventory())
    cohorts, splits, ages, hospitals = audit_cohorts()
    for name, rows in (("cohort_counts", cohorts), ("split_counts", splits),
                       ("age_counts", ages), ("hospital_counts", hospitals),
                       ("historical_grid_comparison", audit_old_results())):
        write_csv(out / f"{name}.csv", rows)
    status_path = out / "revision_status.json"
    status = json.loads(status_path.read_text())
    st = status["stages"]["data_audit"]
    st.update(status="complete", commands=[f"python scripts/pnas_revision/audit.py --run-dir {args.run_dir}"],
              outputs=[f"{name}.csv" for name in ("input_inventory", "cohort_counts", "split_counts", "age_counts", "hospital_counts", "historical_grid_comparison")],
              next_command="review audit outputs and write analysis_protocol.json")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(f"审计完成：{out}")


if __name__ == "__main__":
    main()
