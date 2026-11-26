"""按 eICU 诊断实际录入偏移时间构造截点前诊断分层。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/eicu/eicu-collaborative-research-database-2.0"
CATEGORIES = {
    "infection": ("sepsis", "infection", "pneumonia", "bacteremia", "uti"),
    "cardiovascular": ("cardiac", "heart", "coronary", "myocardial", "arrhythmia", "cardiovascular"),
    "respiratory": ("respiratory", "pulmonary", "ards", "copd", "asthma"),
    "renal": ("renal", "kidney", "aki", "ckd"),
}


def read_events() -> pd.DataFrame:
    ids = set(pd.read_parquet(ROOT / "data/processed/eicu_24h.parquet", columns=["stay_id"]).stay_id)
    pieces = []
    for name, offset, text_cols in (
        ("diagnosis", "diagnosisoffset", ["diagnosisstring"]),
        ("admissionDx", "admitdxenteredoffset", ["admitdxpath", "admitdxname", "admitdxtext"]),
    ):
        for chunk in pd.read_csv(RAW / f"{name}.csv.gz", usecols=["patientunitstayid", offset, *text_cols],
                                 chunksize=200_000, low_memory=False):
            chunk = chunk.loc[chunk.patientunitstayid.isin(ids)].copy()
            chunk["entry_offset_min"] = pd.to_numeric(chunk[offset], errors="coerce")
            chunk = chunk.loc[chunk.entry_offset_min.notna() & chunk.entry_offset_min.lt(1440)]
            if chunk.empty:
                continue
            text = chunk[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
            small = chunk[["patientunitstayid", "entry_offset_min"]].rename(
                columns={"patientunitstayid": "stay_id"})
            small["source_table"] = name
            for category, words in CATEGORIES.items():
                small[category] = text.str.contains("|".join(words), regex=True).to_numpy()
            pieces.append(small)
    return pd.concat(pieces, ignore_index=True)


def build() -> None:
    events = read_events()
    ids = pd.read_parquet(ROOT / "data/processed/eicu_24h.parquet", columns=["stay_id"])
    outputs = []
    for window in (3, 6, 12, 18, 24):
        d = events.loc[events.entry_offset_min.lt(window * 60)].copy()
        grouped = d.groupby("stay_id")
        flags = grouped[list(CATEGORIES)].max().reset_index()
        counts = grouped.agg(n_recorded=("entry_offset_min", "size"),
                             first_entry_offset_min=("entry_offset_min", "min"),
                             last_entry_offset_min=("entry_offset_min", "max")).reset_index()
        frame = ids.merge(flags, on="stay_id", how="left", validate="one_to_one")
        frame = frame.merge(counts, on="stay_id", how="left", validate="one_to_one")
        frame["n_recorded"] = frame.n_recorded.fillna(0).astype(int)
        frame["not_recorded"] = frame.n_recorded.eq(0)
        for category in CATEGORIES:
            frame[category] = frame[category].fillna(False).astype(bool)
        frame["other_recorded"] = frame.n_recorded.gt(0) & ~frame[list(CATEGORIES)].any(axis=1)
        frame["window_h"] = window
        outputs.append(frame)
        print(window, "recorded", int((~frame.not_recorded).sum()), "unrecorded", int(frame.not_recorded.sum()), flush=True)
    out = pd.concat(outputs, ignore_index=True)
    path = ROOT / "data/processed/eicu_fig2_diagnosis.parquet"
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    report = {"source_tables": {"diagnosis": "diagnosisoffset", "admissionDx": "admitdxenteredoffset"},
              "cutoff_rule": "entry offset < window_h*60; negative offset means entered before ICU admission",
              "future_fields_excluded": ["activeupondischarge"],
              "unrecorded_rule": "no diagnosis entered by cutoff, not absence of disease",
              "categories": CATEGORIES, "rows": len(out),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "overlap": "named categories are multi-label and can overlap"}
    (ROOT / "data/processed/eicu_fig2_diagnosis_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("complete", len(out), flush=True)
