"""eICU-CRD cohort builder.

Reads raw eICU CSV/gz files, applies the same inclusion criteria as
mimic_cohort.py, extracts identical variable names, and writes parquet
files per time window. Used for external validation (experiments 2, 4).
"""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from .feature_groups import LABEL_COLS


# ---------------------------------------------------------------------------
# eICU variable name mappings
# ---------------------------------------------------------------------------

# vitalPeriodic column → our feature name
VITAL_COLS = {
    "heartrate":           "hr",
    "systemicsystolic":    "sbp",
    "systemicdiastolic":   "dbp",
    "temperature":         "temp",
    "respiration":         "rr",
    "sao2":                "spo2",
}

# lab labname → our feature name
LAB_NAMES = {
    "creatinine":          "creatinine",
    "lactate":             "lactate",
    "WBC x 1000":          "wbc",
    "Hgb":                 "hgb",
    "sodium":              "sodium",
    "potassium":           "potassium",
    "total bilirubin":     "bilirubin",
    "PT - INR":            "inr",
    "pH":                  "abg_ph",
    "pO2":                 "abg_po2",
    "troponin - T":        "troponin",
    "troponin I":          "troponin",
    "Blood Culture":       "blood_culture",
}

# Antibiotic drug name substrings in medication table
ANTIBIOTIC_KEYWORDS = [
    "penicillin", "cefazolin", "ceftriaxone", "vancomycin",
    "piperacillin", "meropenem", "ciprofloxacin", "metronidazole",
    "azithromycin", "levofloxacin", "ampicillin", "clindamycin",
]

# Vasopressor drug names in medication table
VASOPRESSOR_KEYWORDS = [
    "norepinephrine", "epinephrine", "dopamine", "vasopressin", "phenylephrine",
]

# Diagnosis category keywords in diagnosis table (diagnosisstring)
DX_INFECTION_KEYWORDS = ["sepsis", "infection", "pneumonia", "uti", "bacteremia"]
DX_CARDIO_KEYWORDS = ["cardiac", "heart", "coronary", "myocardial", "arrhythmia"]
DX_RESP_KEYWORDS = ["respiratory", "pulmonary", "ards", "copd", "asthma"]
DX_RENAL_KEYWORDS = ["renal", "kidney", "aki", "ckd"]


def _read_csv(path: Path) -> pd.DataFrame:
    if str(path).endswith(".gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _find(eicu_dir: Path, name: str) -> Path:
    for suffix in ["", ".csv", ".csv.gz"]:
        p = eicu_dir / (name + suffix)
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {name} in {eicu_dir}")


def _read_csv_filtered(
    path: Path,
    filter_col: str,
    filter_ids: set,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    chunks = []
    reader = pd.read_csv(
        path,
        compression="gzip" if str(path).endswith(".gz") else None,
        low_memory=False,
        chunksize=chunksize,
    )
    for chunk in reader:
        if filter_col in chunk.columns:
            sub = chunk[chunk[filter_col].isin(filter_ids)]
            if len(sub) > 0:
                chunks.append(sub)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _flush_shards(buffers: dict, shard_dir: Path) -> None:
    for gid, dfs in buffers.items():
        shard_path = shard_dir / f"{int(gid)}.parquet"
        new_df = pd.concat(dfs, ignore_index=True)
        if shard_path.exists():
            existing = pd.read_parquet(shard_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_parquet(shard_path, index=False)


def _read_shard(shard_dir: Path, gid: int) -> pd.DataFrame:
    p = shard_dir / f"{int(gid)}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    q = shard_dir.parent.parent / "cache_eicu_landmark_visit" / shard_dir.name / f"{int(gid)}.parquet"
    return pd.read_parquet(q) if q.exists() else pd.DataFrame()


def _extract_all_windows(arg):
    """一位患者只读取一次三个事件 shard，计算所有合格截点。"""
    stay, time_windows = arg
    cache_dir = Path(__file__).resolve().parents[2] / 'data/processed/cache_eicu'
    sid = stay['patientunitstayid']
    lab = _read_shard(cache_dir / 'lab', sid)
    vital = _read_shard(cache_dir / 'vitalPeriodic', sid)
    medication = _read_shard(cache_dir / 'medication', sid)
    result = []
    for window_h in time_windows:
        if stay['unitdischargeoffset'] > window_h * 60:
            result.append((window_h, _extract_stay_features(
                stay, window_h, lab, vital, medication, {}, pd.DataFrame())))
    return result


def build_cohort(
    eicu_dir: str,
    time_windows: list[int],
    output_dir: str,
    max_stays: int | None = None,
    chunk_size: int = 200_000,
) -> None:
    """Build eICU cohort and write feature parquets per time window.

    Two-pass design: scan each large file once to build per-stay parquet shards,
    then read shards during feature extraction (no full-table RAM load).

    Args:
        eicu_dir: root directory of eICU-CRD
        time_windows: list of window sizes in hours
        output_dir: directory to write output parquet files
        max_stays: if set, limit to this many stays (for smoke tests)
        chunk_size: rows per chunk when scanning large CSVs
    """
    import gc
    eicu_dir   = Path(eicu_dir)
    output_dir = Path(output_dir)
    cache_dir = Path(__file__).resolve().parents[2] / "data/processed/cache_eicu"
    cache_dir.mkdir(parents=True, exist_ok=True)
    short_cache = cache_dir.parent / "cache_eicu_landmark_visit"

    # Small tables — load fully
    print("Loading eICU small tables...")
    patient   = _read_csv(_find(eicu_dir, "patient"))
    apache = None
    diagnosis = None  # 最终诊断不进入正式预测数据

    # 同一患者先按明确的 ICU 访次排序，再判定年龄和各截点风险资格。
    patient = patient.sort_values(["uniquepid", "unitvisitnumber", "patientunitstayid"]).drop_duplicates(
        subset="uniquepid", keep="first"
    ).copy()
    patient = patient.dropna(subset=["unitdischargeoffset"])
    patient["los_h"] = patient["unitdischargeoffset"] / 60
    patient["age_num"] = pd.to_numeric(
        patient["age"].astype(str).str.replace("> ", "").str.replace(">", ""),
        errors="coerce",
    )
    patient = patient[patient["age_num"] >= 18].reset_index(drop=True)
    if max_stays is not None:
        patient = patient.head(max_stays)
    print(f"eICU cohort size: {len(patient)}")

    legacy_ids = set(pd.read_parquet(cache_dir.parent / "eicu_12h.parquet", columns=["stay_id"]).stay_id)
    cohort_ids = set(patient.patientunitstayid) - legacy_ids
    apache_sofa = {}

    diag_by_stay = {}
    del apache, diagnosis
    gc.collect()

    # Large tables — scan once, shard by patientunitstayid
    large_files = {
        "lab":           _find(eicu_dir, "lab"),
        "vitalPeriodic": _find(eicu_dir, "vitalPeriodic"),
        "medication":    _find(eicu_dir, "medication"),
    }

    for table_name, csv_path in large_files.items():
        shard_dir = short_cache / table_name
        if (shard_dir / ".complete").exists():
            print(f"  {table_name}: short-stay cache complete, skipping scan")
            continue
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Scanning {table_name} ({csv_path.name})...")
        buffers: dict[int, list] = {}
        buf_rows = 0
        FLUSH_ROWS = 2_000_000
        reader = pd.read_csv(
            csv_path,
            compression="gzip" if str(csv_path).endswith(".gz") else None,
            low_memory=False,
            chunksize=chunk_size,
        )
        for chunk in reader:
            if "patientunitstayid" not in chunk.columns:
                continue
            sub = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if sub.empty:
                continue
            for gid, grp in sub.groupby("patientunitstayid"):
                buffers.setdefault(gid, []).append(grp)
            buf_rows += len(sub)
            if buf_rows >= FLUSH_ROWS:
                _flush_shards(buffers, shard_dir)
                buffers = {}
                buf_rows = 0
                gc.collect()
        if buffers:
            _flush_shards(buffers, shard_dir)
        (shard_dir / ".complete").write_text("complete\n")
        print(f"    done → {shard_dir}")
        gc.collect()

    # 多进程计算按患者独立执行；每位患者只读取一次事件缓存。
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {window: [] for window in time_windows}
    args = ((stay, tuple(time_windows)) for _, stay in patient.iterrows())
    with ProcessPoolExecutor(max_workers=8) as pool:
        for result in tqdm(pool.map(_extract_all_windows, args, chunksize=16),
                           total=len(patient), mininterval=10):
            for window_h, row in result:
                records[window_h].append(row)
    for window_h, rows in records.items():
        out_path = output_dir / f"eicu_{window_h}h.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"Saved {len(rows)} stays → {out_path}", flush=True)


def _extract_stay_features(
    stay: pd.Series,
    window_h: int,
    lab: pd.DataFrame,
    vital: pd.DataFrame,
    medication: pd.DataFrame,
    apache_sofa: dict,
    diagnosis: pd.DataFrame,
) -> dict:
    """All DataFrames are pre-filtered to this stay only."""
    stay_id = stay["patientunitstayid"]
    window_min = window_h * 60  # convert to minutes

    # Ensure expected columns exist on empty DataFrames to avoid KeyError
    if lab.empty:
        lab = pd.DataFrame(columns=["labresultoffset", "labresultrevisedoffset", "available_offset", "labname", "labresult"])
    if vital.empty:
        vital = pd.DataFrame(columns=["observationoffset", "systemicmean"] + list(VITAL_COLS.keys()))
    if medication.empty:
        medication = pd.DataFrame(columns=["drugstartoffset", "drugname"])

    row: dict = {
        "stay_id": stay_id,
        "subject_id": stay["uniquepid"],
        "hospital_id": stay.get("hospitalid", np.nan),
        "window_h": window_h,
    }

    # --- Disease-state features ---
    row["age"] = float(stay["age_num"])
    row["gender"] = 1 if str(stay.get("gender", "")).upper() in ("MALE", "M") else 0
    row["admission_source"] = _encode_admission_source(str(stay.get("unitadmitsource", "")))

    # Vital sign first values within window (offset in minutes)
    vit_stay = vital[
        (vital["observationoffset"] >= 0) &
        (vital["observationoffset"] < window_min)
    ].sort_values("observationoffset") if not vital.empty else vital

    for eicu_col, feat in VITAL_COLS.items():
        if eicu_col in vit_stay.columns:
            sub = vit_stay[eicu_col].dropna()
            value = float(sub.iloc[0]) if len(sub) > 0 else np.nan
            if feat == "temp" and 80 <= value <= 115:
                value = (value - 32) * 5 / 9
            row[f"{feat}_first"] = value
        else:
            row[f"{feat}_first"] = np.nan

    # Lab first values within window
    lab = lab.copy()
    if not lab.empty:
        lab["available_offset"] = pd.to_numeric(lab["labresultoffset"], errors="coerce")
        if "labresultrevisedoffset" in lab.columns:
            revised = pd.to_numeric(lab["labresultrevisedoffset"], errors="coerce")
            lab["available_offset"] = np.maximum(lab["available_offset"], revised.fillna(lab["available_offset"]))
    lab_stay = lab[
        (lab["labresultoffset"] >= 0) &
        (lab["available_offset"] < window_min)
    ] if not lab.empty else lab
    for lab_name, feat in [
        ("creatinine", "creatinine"), ("lactate", "lactate"), ("WBC x 1000", "wbc"),
        ("Hgb", "hgb"), ("sodium", "sodium"), ("potassium", "potassium"),
        ("total bilirubin", "bilirubin"), ("PT - INR", "inr"),
    ]:
        sub = lab_stay[lab_stay["labname"] == lab_name].sort_values("available_offset")
        val = pd.to_numeric(sub["labresult"], errors="coerce").dropna()
        row[f"{feat}_first"] = float(val.iloc[0]) if len(val) > 0 else np.nan

    # --- Observation-behaviour features ---
    for wh in [3, 6, 12, 18, 24]:
        wmin = wh * 60
        if wh > window_h:
            row[f"vitals_count_{wh}h"] = np.nan
            row[f"labs_count_{wh}h"] = np.nan
            row[f"creatinine_repeat_{wh}h"] = np.nan
            row[f"lactate_repeat_{wh}h"] = np.nan
            if wh == 6:
                row["has_lactate_6h"] = np.nan
                row["has_creatinine_6h"] = np.nan
                row["orders_count_6h"] = np.nan
            if wh == 24:
                for k in ["abg_count_24h", "orders_count_24h", "has_blood_culture_24h",
                          "has_abg_24h", "has_troponin_24h", "has_cxr_24h", "cxr_count_24h"]:
                    row[k] = np.nan
            continue

        vit_w = vital[
            (vital["observationoffset"] >= 0) &
            (vital["observationoffset"] < wmin)
        ] if not vital.empty else vital
        lab_w = lab_stay[lab_stay["available_offset"] < wmin]

        row[f"vitals_count_{wh}h"] = int(len(vit_w))
        row[f"labs_count_{wh}h"] = int(len(lab_w))

        cr_w = lab_w[lab_w["labname"] == "creatinine"]
        row[f"creatinine_repeat_{wh}h"] = max(0, len(cr_w) - 1)
        lact_w = lab_w[lab_w["labname"] == "lactate"]
        row[f"lactate_repeat_{wh}h"] = max(0, len(lact_w) - 1)

        if wh == 6:
            row["has_lactate_6h"] = int(len(lact_w) > 0)
            row["has_creatinine_6h"] = int(len(cr_w) > 0)
            med_w = medication[
                (medication["drugstartoffset"] >= 0) &
                (medication["drugstartoffset"] < wmin) &
                (medication["drugorderoffset"] < wmin)
            ] if not medication.empty else medication
            row["orders_count_6h"] = int(len(med_w))

    lab_24 = lab_stay[lab_stay["available_offset"] < min(1440, window_min)]
    abg_lab = lab_24[lab_24["labname"].isin(["pH", "pO2", "pCO2"])]
    row["abg_count_24h"] = int(len(abg_lab))
    row["has_abg_24h"] = int(len(abg_lab) > 0)
    row["has_blood_culture_24h"] = int(
        len(lab_24[lab_24["labname"].str.lower().str.contains("blood culture", na=False)]) > 0
    )
    row["has_troponin_24h"] = int(
        len(lab_24[lab_24["labname"].str.lower().str.contains("troponin", na=False)]) > 0
    )
    # eICU has no separate imaging table; proxy via nurseCharting or set 0
    row["has_cxr_24h"] = 0
    row["cxr_count_24h"] = 0

    # 24h 命名为历史兼容；治疗与处方只使用预测截点之前的信息。
    med_24 = medication[
        (medication["drugstartoffset"] >= 0) &
        (medication["drugstartoffset"] < min(1440, window_min)) &
        (medication["drugorderoffset"] < min(1440, window_min))
    ] if not medication.empty else medication
    row["orders_count_24h"] = int(len(med_24))

    # --- Mixed features (24h) ---
    drug_names = med_24["drugname"].str.lower().fillna("")
    row["antibiotics_24h"] = int(
        drug_names.str.contains("|".join(ANTIBIOTIC_KEYWORDS), na=False).any()
    )
    row["vasopressors_24h"] = int(
        drug_names.str.contains("|".join(VASOPRESSOR_KEYWORDS), na=False).any()
    )
    # Mechanical ventilation: check medication for "mechanical ventilation" or ventilator
    row["mechanical_ventilation_24h"] = int(
        drug_names.str.contains("ventilat", na=False).any()
    )
    row["blood_culture_sent_24h"] = row["has_blood_culture_24h"]

    # --- Labels ---
    row.update(_extract_labels(stay, stay_id, lab, vital, medication, window_min))
    return row


def _compute_sepsis_deterioration(vital: pd.DataFrame, window_min: int, discharge_offset: float) -> float:
    """MAP drop below 65 in the 24h after window (eICU lacks a SOFA timeline,
    so we fall back to a MAP-only proxy — weaker than the MIMIC GCS+MAP proxy
    but keeps both classes present)."""
    if vital.empty or "systemicmean" not in vital.columns:
        return 0
    pre_start = max(0, window_min - 360)
    post_end = window_min + 1440
    pre_map = vital[
        (vital["observationoffset"] >= pre_start) &
        (vital["observationoffset"] < window_min)
    ]["systemicmean"]
    pre_map = pd.to_numeric(pre_map, errors="coerce").dropna()
    post_map = vital[
        (vital["observationoffset"] >= window_min) &
        (vital["observationoffset"] < post_end)
    ]["systemicmean"]
    post_map = pd.to_numeric(post_map, errors="coerce").dropna()
    if len(pre_map) > 0 and len(post_map) > 0:
        if float(pre_map.median()) >= 70 and float(post_map.min()) < 65:
            return 1
    return 0.0 if len(pre_map) and len(post_map) and discharge_offset >= post_end else np.nan


def _extract_labels(
    stay: pd.Series,
    stay_id: int,
    lab: pd.DataFrame,
    vital: pd.DataFrame,
    medication: pd.DataFrame,
    window_min: int,
) -> dict:
    """All DataFrames are pre-filtered to this stay only."""
    unit_status = str(stay.get("unitdischargestatus", "")).lower()
    hosp_status = str(stay.get("hospitaldischargestatus", "")).lower()
    icu_mort = int(unit_status == "expired") if unit_status in ("expired", "alive") else np.nan
    hosp_mort = int(hosp_status == "expired") if hosp_status in ("expired", "alive") else np.nan
    inhosp_mort = hosp_mort if pd.notna(stay.get("hospitaldischargeoffset")) and stay["hospitaldischargeoffset"] > window_min else np.nan

    cr_all = lab[lab["labname"] == "creatinine"].copy() if not lab.empty else lab
    aki = np.nan
    if len(cr_all):
        cr_all["labresult"] = pd.to_numeric(cr_all["labresult"], errors="coerce")
        cr_all = cr_all.dropna(subset=["labresult"]).sort_values("labresultoffset")
        pre = cr_all[(cr_all["available_offset"] >= 0) & (cr_all["available_offset"] < window_min)]
        post = cr_all[(cr_all["labresultoffset"] > window_min) & (cr_all["labresultoffset"] <= window_min + 2880)]
        if len(pre) and len(post):
            baseline = float(pre.iloc[-1]["labresult"])
            peak = float(post.labresult.max())
            if (peak - baseline >= 0.3) or (peak >= 1.5 * baseline):
                aki = 1.0
            elif stay["unitdischargeoffset"] >= window_min + 2880:
                aki = 0.0

    sep_det = _compute_sepsis_deterioration(vital, window_min, stay["unitdischargeoffset"])

    med_post = medication[(medication["drugstartoffset"] > window_min) & (medication["drugstartoffset"] <= window_min + 1440)] if not medication.empty else medication
    drug_names = med_post["drugname"].str.lower().fillna("") if not med_post.empty else pd.Series([], dtype=str)
    vaso_post = drug_names.str.contains("|".join(VASOPRESSOR_KEYWORDS), na=False).any()
    mv_post   = drug_names.str.contains("ventilat", na=False).any()
    esc = 1.0 if vaso_post or mv_post else (0.0 if stay["unitdischargeoffset"] >= window_min + 1440 else np.nan)

    return {
        "icu_mortality":        icu_mort,
        "inhosp_mortality":     inhosp_mort,
        "aki":                  aki,
        "sepsis_deterioration": sep_det,
        "escalation":           esc,
    }


def _compute_apache_proxy_sofa(apache: pd.DataFrame) -> dict[int, float]:
    """Map APACHE IV score to a proxy SOFA score (linear: APACHE/5, capped at 24)."""
    if "apachescore" not in apache.columns:
        return {}
    valid = apache[["patientunitstayid", "apachescore"]].dropna()
    valid = valid.drop_duplicates(subset="patientunitstayid", keep="first")
    valid["sofa_proxy"] = (valid["apachescore"] / 5).clip(upper=24)
    return valid.set_index("patientunitstayid")["sofa_proxy"].to_dict()


def _encode_admission_source(loc: str) -> int:
    loc = loc.lower()
    if "emergency" in loc or "ed" in loc:
        return 1
    if "transfer" in loc or "other hospital" in loc:
        return 2
    if "elective" in loc or "operating" in loc:
        return 3
    return 0
