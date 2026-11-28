"""MIMIC-IV cohort builder.

Reads raw MIMIC-IV CSV/gz files, applies inclusion criteria, extracts
disease-state, observation-behaviour, and mixed features for each ICU stay,
then writes a single parquet file per time window.
"""

from __future__ import annotations

import gzip
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .feature_groups import LABEL_COLS

# ---------------------------------------------------------------------------
# MIMIC-IV item IDs
# ---------------------------------------------------------------------------

# chartevents item IDs for vital signs
VITAL_ITEMIDS = {
    "hr":   [220045],
    "sbp":  [220179, 220050],
    "dbp":  [220180, 220051],
    "temp": [223761, 223762, 226329],
    "rr":   [220210, 224690],
    "spo2": [220277],
}

# labevents item IDs
LAB_ITEMIDS = {
    "creatinine": [50912],
    "lactate":    [50813],
    "wbc":        [51301],
    "hgb":        [51222],
    "sodium":     [50983],
    "potassium":  [50971],
    "bilirubin":  [50885],
    "inr":        [51237],
    "abg":        [50820, 50821, 50818],   # pH, pO2, pCO2
    "troponin":   [51002, 52598],
    "blood_culture": [90109],
}

# ICD-10 chapter prefixes for diagnosis categories
DX_MAP = {
    "dx_infection":      ["A", "B"],
    "dx_cardiovascular": ["I"],
    "dx_respiratory":    ["J"],
    "dx_renal":          ["N"],
}

# Comorbidity ICD prefixes (simplified Charlson mapping)
COMORBIDITY_MAP = {
    "has_diabetes":      ["E10", "E11", "E12", "E13", "E14"],
    "has_ckd":           ["N18"],
    "has_heart_failure": ["I50"],
    "has_copd":          ["J44"],
    "has_cancer":        ["C"],
}

# Vasopressor item IDs in inputevents
VASOPRESSOR_ITEMIDS = [221906, 221289, 221749, 222315, 221662]
# Mechanical ventilation in procedureevents
MV_ITEMIDS = [225792, 225794]
# Antibiotic drug names (substring match in prescriptions)
ANTIBIOTIC_KEYWORDS = ["penicillin", "cefazolin", "ceftriaxone", "vancomycin",
                       "piperacillin", "meropenem", "ciprofloxacin", "metronidazole",
                       "azithromycin", "levofloxacin", "ampicillin", "clindamycin"]


def _read_csv(path: Path) -> pd.DataFrame:
    if str(path).endswith(".gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _read_csv_filtered(
    path: Path,
    filter_col: str,
    filter_ids: set,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Read a large CSV/gz in chunks, keeping only rows where filter_col is in filter_ids."""
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


def _parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


POE_ORDER_TIMES = None


def _extract_all_windows(arg):
    """一位患者只读取一次原始事件 shard，计算所有合格截点。"""
    stay, time_windows = arg
    cache_dir = Path(__file__).resolve().parents[2] / 'data/processed/cache'
    sid, hid = stay['stay_id'], stay['hadm_id']
    ce = _read_shard(cache_dir / 'chartevents', id_col='stay_id', gid=sid)
    le = _read_shard(cache_dir / 'labevents', id_col='hadm_id', gid=hid)
    pr = _read_shard(cache_dir / 'prescriptions', id_col='hadm_id', gid=hid)
    ie = _read_shard(cache_dir / 'inputevents', id_col='stay_id', gid=sid)
    pe = _read_shard(cache_dir / 'procedureevents', id_col='stay_id', gid=sid)
    if not pr.empty:
        pr['order_time'] = _parse_time(pr['poe_id'].map(POE_ORDER_TIMES))
    if not ce.empty:
        ce['charttime'] = _parse_time(ce['charttime'])
        ce['valuenum'] = pd.to_numeric(ce['valuenum'], errors='coerce')
        ce['storetime'] = _parse_time(ce['storetime'])
        ce['available_time'] = ce[['charttime', 'storetime']].max(axis=1).where(ce['storetime'].notna())
    if not le.empty:
        le['charttime'] = _parse_time(le['charttime'])
        le['valuenum'] = pd.to_numeric(le['valuenum'], errors='coerce')
        le['storetime'] = _parse_time(le['storetime'])
        le['available_time'] = le[['charttime', 'storetime']].max(axis=1).where(le['storetime'].notna())
    for events in (pr, ie, pe):
        if not events.empty and 'starttime' in events.columns:
            events['starttime'] = _parse_time(events['starttime'])
            if 'storetime' in events.columns:
                events['storetime'] = _parse_time(events['storetime'])
    result = []
    for window_h in time_windows:
        cutoff = stay['intime'] + pd.Timedelta(hours=window_h)
        if pd.isna(stay['outtime']) or stay['outtime'] <= cutoff:
            continue
        death_time = stay.get('deathtime')
        if pd.notna(death_time) and death_time <= cutoff:
            continue
        dod = stay.get('dod')
        if pd.isna(death_time) and pd.notna(dod) and dod.normalize() <= cutoff.normalize():
            continue
        result.append((window_h, _extract_stay_features(
            stay, window_h, ce, le, pr, ie, pe, {}, {}, {})))
    return result


def build_cohort(
    mimic_dir: str,
    time_windows: list[int],
    output_dir: str,
    max_stays: int | None = None,
    chunk_size: int = 200_000,
) -> None:
    """Build MIMIC-IV ICU cohort and write feature parquets.

    Two-pass design to minimise I/O and RAM:
      Pass 1 (once per large file): scan CSV → write per-stay parquet shards
              to cache/.  Each large file is read exactly once regardless of
              how many time windows are requested.
      Pass 2 (per time window):    read shards from cache, extract features,
              append to output parquet.

    Args:
        mimic_dir: root directory of MIMIC-IV (contains hosp/ and icu/)
        time_windows: list of window sizes in hours, e.g. [6, 12, 24]
        output_dir: directory to write output parquet files
        max_stays: if set, only process this many stays (smoke tests)
        chunk_size: rows per chunk when scanning large CSVs
    """
    import gc
    mimic_dir  = Path(mimic_dir)
    output_dir = Path(output_dir)
    cache_dir  = Path(__file__).resolve().parents[2] / "data/processed/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    short_cache = cache_dir.parent / "cache_landmark_short"

    # ------------------------------------------------------------------
    # Step 1: small lookup tables (all easily fit in RAM)
    # ------------------------------------------------------------------
    print("Loading small lookup tables...")
    icustays   = _read_csv(_find(mimic_dir, "icu/icustays"))
    patients   = _read_csv(_find(mimic_dir, "hosp/patients"))
    admissions = _read_csv(_find(mimic_dir, "hosp/admissions"))
    diagnoses = None  # 最终诊断不进入正式预测数据

    icustays["intime"]  = _parse_time(icustays["intime"])
    icustays["outtime"] = _parse_time(icustays["outtime"])
    icustays = icustays.sort_values(["subject_id", "intime"])
    icustays = icustays.drop_duplicates(subset="subject_id", keep="first")

    patients["anchor_age"] = patients["anchor_age"].astype(float)
    patients["dod"] = _parse_time(patients["dod"])
    icustays = icustays.merge(
        patients[["subject_id", "anchor_age", "gender", "anchor_year", "dod"]],
        on="subject_id", how="left",
    )
    admissions["admittime"] = _parse_time(admissions["admittime"])
    admissions["deathtime"] = _parse_time(admissions["deathtime"])
    admissions["dischtime"] = _parse_time(admissions["dischtime"])
    icustays = icustays.merge(
        admissions[["hadm_id", "admittime", "admission_location", "insurance",
                     "discharge_location", "hospital_expire_flag", "deathtime", "dischtime"]],
        on="hadm_id", how="left",
    )
    icustays["los_h"] = (
        (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600
    )
    icustays = icustays[icustays["anchor_age"] >= 18].copy()
    icustays = icustays.reset_index(drop=True)
    if max_stays is not None:
        icustays = icustays.head(max_stays)
    print(f"Cohort size: {len(icustays)}")

    short_stays = icustays[icustays["los_h"] < 12]
    cohort_stay_ids = set(short_stays["stay_id"].tolist())
    cohort_hadm_ids = set(short_stays["hadm_id"].tolist())

    comorbidity_df, dx_cat_df = {}, {}
    del diagnoses, patients, admissions
    gc.collect()

    poe_path = cache_dir.parent / "poe_order_times.parquet"
    if not poe_path.exists():
        raise FileNotFoundError(f"缺少处方录入时间索引: {poe_path}")
    global POE_ORDER_TIMES
    POE_ORDER_TIMES = pd.read_parquet(poe_path).set_index("poe_id")["ordertime"]

    # ------------------------------------------------------------------
    # Step 2: scan each large file ONCE → per-stay parquet shards in cache
    # ------------------------------------------------------------------
    large_files = {
        "chartevents":     (_find(mimic_dir, "icu/chartevents"),     "stay_id"),
        "labevents":       (_find(mimic_dir, "hosp/labevents"),      "hadm_id"),
        "prescriptions":   (_find(mimic_dir, "hosp/prescriptions"),  "hadm_id"),
        "inputevents":     (_find(mimic_dir, "icu/inputevents"),     "stay_id"),
        "procedureevents": (_find(mimic_dir, "icu/procedureevents"), "stay_id"),
    }

    for table_name, (csv_path, id_col) in large_files.items():
        shard_dir = short_cache / table_name
        if (shard_dir / ".complete").exists():
            print(f"  {table_name}: short-stay cache complete, skipping scan")
            continue
        shard_dir.mkdir(parents=True, exist_ok=True)
        filter_ids = cohort_stay_ids if id_col == "stay_id" else cohort_hadm_ids
        print(f"  Scanning {table_name} ({csv_path.name})...")
        buffers: dict[int, list[pd.DataFrame]] = {}
        buf_rows = 0
        FLUSH_ROWS = 2_000_000  # flush accumulated rows to parquet every 2M rows

        reader = pd.read_csv(
            csv_path,
            compression="gzip" if str(csv_path).endswith(".gz") else None,
            low_memory=False,
            chunksize=chunk_size,
        )
        for chunk in reader:
            if id_col not in chunk.columns:
                continue
            sub = chunk[chunk[id_col].isin(filter_ids)]
            if sub.empty:
                continue
            for gid, grp in sub.groupby(id_col):
                buffers.setdefault(gid, []).append(grp)
            buf_rows += len(sub)
            if buf_rows >= FLUSH_ROWS:
                _flush_shards(buffers, shard_dir, id_col)
                buffers = {}
                buf_rows = 0
                gc.collect()
        if buffers:
            _flush_shards(buffers, shard_dir, id_col)
        (shard_dir / ".complete").write_text("complete\n")
        print(f"    done → {shard_dir}")
        gc.collect()

    # ------------------------------------------------------------------
    # Step 3: SOFA lookup — read from chartevents shard cache
    # ------------------------------------------------------------------
    sofa_dict = {}  # 整次住院 SOFA 不进入正式预测数据

    # 多进程计算按患者独立执行；每位患者只读取一次缓存。
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {window: [] for window in time_windows}
    args = ((stay, tuple(time_windows)) for _, stay in icustays.iterrows())
    with ProcessPoolExecutor(max_workers=8,
                             mp_context=multiprocessing.get_context('fork')) as pool:
        for result in tqdm(pool.map(_extract_all_windows, args, chunksize=16),
                           total=len(icustays), mininterval=10):
            for window_h, row in result:
                records[window_h].append(row)
    for window_h, rows in records.items():
        out_path = output_dir / f"mimic_iv_{window_h}h.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"Saved {len(rows)} stays → {out_path}", flush=True)


def _flush_shards(
    buffers: dict[int, list[pd.DataFrame]],
    shard_dir: Path,
    id_col: str,
) -> None:
    """Append accumulated rows for each ID to its parquet shard file."""
    for gid, dfs in buffers.items():
        shard_path = shard_dir / f"{int(gid)}.parquet"
        new_df = pd.concat(dfs, ignore_index=True)
        if shard_path.exists():
            existing = pd.read_parquet(shard_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_parquet(shard_path, index=False)


def _read_shard(shard_dir: Path, id_col: str, gid: int) -> pd.DataFrame:
    """Read the cached parquet shard for a single stay/hadm ID."""
    p = shard_dir / f"{int(gid)}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    q = shard_dir.parent.parent / "cache_landmark_short" / shard_dir.name / f"{int(gid)}.parquet"
    return pd.read_parquet(q) if q.exists() else pd.DataFrame()


def _compute_sofa_from_cache(
    chartevents_cache: Path,
    icustays: pd.DataFrame,
) -> dict[int, float]:
    """Compute a SOFA proxy per stay from cached chartevents shards.

    Tries itemid 227428 (direct SOFA score) first.  If absent, estimates
    neurological SOFA from GCS components (eye 220739, verbal 223900,
    motor 223901): GCS total = sum of three; neuro_sofa = 4 if GCS<=5,
    3 if <=9, 2 if <=12, 1 if <=14, 0 if 15.  This gives a valid [0,4]
    sub-score that correlates with severity even when full SOFA is missing.
    """
    SOFA_DIRECT  = {227428}
    GCS_EYE      = {220739}
    GCS_VERBAL   = {223900}
    GCS_MOTOR    = {223901}

    result = {}
    for _, stay in icustays.iterrows():
        sid = stay["stay_id"]
        ce = _read_shard(chartevents_cache, "stay_id", sid)
        if ce.empty or "itemid" not in ce.columns:
            continue
        ce_t = ce.copy()
        ce_t["charttime"] = _parse_time(ce_t["charttime"])
        ce_t["valuenum"]  = pd.to_numeric(ce_t["valuenum"], errors="coerce")
        ce_t = ce_t.sort_values("charttime")

        # Try direct SOFA score
        direct = ce_t[ce_t["itemid"].isin(SOFA_DIRECT)]["valuenum"].dropna()
        if len(direct) > 0:
            result[sid] = float(direct.iloc[0])
            continue

        # Fall back to GCS-based neuro SOFA proxy
        def _first_val(item_set):
            sub = ce_t[ce_t["itemid"].isin(item_set)]["valuenum"].dropna()
            return float(sub.iloc[0]) if len(sub) > 0 else np.nan

        eye    = _first_val(GCS_EYE)
        verbal = _first_val(GCS_VERBAL)
        motor  = _first_val(GCS_MOTOR)
        if np.isnan(eye) or np.isnan(verbal) or np.isnan(motor):
            continue
        gcs = eye + verbal + motor
        if   gcs <= 5:  neuro = 4.0
        elif gcs <= 9:  neuro = 3.0
        elif gcs <= 12: neuro = 2.0
        elif gcs <= 14: neuro = 1.0
        else:           neuro = 0.0
        result[sid] = neuro
    return result


def _find(mimic_dir: Path, rel: str) -> Path:
    """Find CSV or CSV.gz file by relative path stem."""
    for suffix in ["", ".csv", ".csv.gz"]:
        p = mimic_dir / (rel + suffix)
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {rel} in {mimic_dir}")


def _extract_stay_features(
    stay: pd.Series,
    window_h: int,
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame,
    prescriptions: pd.DataFrame,
    inputevents: pd.DataFrame,
    procedureevents: pd.DataFrame,
    comorbidity_df: pd.DataFrame,
    dx_cat_df: pd.DataFrame,
    sofa_df: dict,
) -> dict:
    stay_id = stay["stay_id"]
    hadm_id = stay["hadm_id"]
    intime = stay["intime"]
    window_end = intime + pd.Timedelta(hours=window_h)

    # Ensure expected columns exist on empty DataFrames to avoid KeyError
    if chartevents.empty:
        chartevents = pd.DataFrame(columns=["charttime", "available_time", "itemid", "valuenum"])
    if labevents.empty:
        labevents = pd.DataFrame(columns=["charttime", "available_time", "itemid", "valuenum"])
    if prescriptions.empty:
        prescriptions = pd.DataFrame(columns=["starttime", "order_time", "drug"])
    if inputevents.empty:
        inputevents = pd.DataFrame(columns=["starttime", "itemid", "amount"])
    if procedureevents.empty:
        procedureevents = pd.DataFrame(columns=["starttime", "itemid"])

    row: dict = {"stay_id": stay_id, "hadm_id": hadm_id,
                 "subject_id": stay["subject_id"], "window_h": window_h}

    # --- Disease-state features ---
    row["age"] = float(stay["anchor_age"])
    row["gender"] = 1 if str(stay["gender"]).upper() == "M" else 0
    row["admission_source"] = _encode_admission_source(stay.get("admission_location", ""))

    # Vital sign first values within window (chartevents already filtered to this stay)
    ce_stay = chartevents[
        (chartevents["charttime"] >= intime) &
        (chartevents["available_time"] < window_end)
    ]
    for vital, itemids in VITAL_ITEMIDS.items():
        sub = ce_stay[ce_stay["itemid"].isin(itemids)].sort_values(["available_time", "charttime"])
        value = float(sub["valuenum"].iloc[0]) if len(sub) > 0 else np.nan
        if vital == "temp" and 80 <= value <= 115:
            value = (value - 32) * 5 / 9
        row[f"{vital}_first"] = value

    # Lab first values within window (labevents already filtered to this hadm)
    le_stay = labevents[
        (labevents["charttime"] >= intime) &
        (labevents["available_time"] < window_end)
    ]
    for lab in ["creatinine", "lactate", "wbc", "hgb", "sodium", "potassium", "bilirubin", "inr"]:
        itemids = LAB_ITEMIDS[lab]
        sub = le_stay[le_stay["itemid"].isin(itemids)].sort_values(["available_time", "charttime"])
        row[f"{lab}_first"] = float(sub["valuenum"].iloc[0]) if len(sub) > 0 else np.nan

    # --- Observation-behaviour features ---
    for wh in [3, 6, 12, 18, 24]:
        if wh > window_h:
            # Fill with NaN for windows larger than current window
            row[f"vitals_count_{wh}h"] = np.nan
            row[f"labs_count_{wh}h"] = np.nan
            row[f"creatinine_repeat_{wh}h"] = np.nan
            row[f"lactate_repeat_{wh}h"] = np.nan
            if wh == 6:
                row["has_lactate_6h"] = np.nan
                row["has_creatinine_6h"] = np.nan
                row["orders_count_6h"] = np.nan
            if wh == 24:
                row["abg_count_24h"] = np.nan
                row["orders_count_24h"] = np.nan
                row["has_blood_culture_24h"] = np.nan
                row["has_abg_24h"] = np.nan
                row["has_troponin_24h"] = np.nan
                row["has_cxr_24h"] = np.nan
                row["cxr_count_24h"] = np.nan
            continue
        wend = intime + pd.Timedelta(hours=wh)
        ce_w = ce_stay[ce_stay["available_time"] < wend]
        le_w = le_stay[le_stay["available_time"] < wend]

        all_vital_ids = [iid for ids in VITAL_ITEMIDS.values() for iid in ids]
        row[f"vitals_count_{wh}h"] = int(ce_w[ce_w["itemid"].isin(all_vital_ids)].shape[0])
        row[f"labs_count_{wh}h"] = int(le_w.shape[0])

        cr_sub = le_w[le_w["itemid"].isin(LAB_ITEMIDS["creatinine"])]
        row[f"creatinine_repeat_{wh}h"] = max(0, len(cr_sub) - 1)
        lact_sub = le_w[le_w["itemid"].isin(LAB_ITEMIDS["lactate"])]
        row[f"lactate_repeat_{wh}h"] = max(0, len(lact_sub) - 1)

        if wh == 6:
            row["has_lactate_6h"] = int(len(lact_sub) > 0)
            row["has_creatinine_6h"] = int(len(cr_sub) > 0)
            row["orders_count_6h"] = _count_orders(prescriptions, intime, wend)

    # 24h-specific observation features
    wend24 = intime + pd.Timedelta(hours=24)
    le_24 = le_stay[le_stay["available_time"] < wend24]
    ce_24 = ce_stay[ce_stay["available_time"] < wend24]

    row["abg_count_24h"] = int(le_24[le_24["itemid"].isin(LAB_ITEMIDS["abg"])].shape[0])
    row["has_blood_culture_24h"] = int(
        le_24[le_24["itemid"].isin(LAB_ITEMIDS["blood_culture"])].shape[0] > 0
    )
    row["has_abg_24h"] = int(row["abg_count_24h"] > 0)
    row["has_troponin_24h"] = int(
        le_24[le_24["itemid"].isin(LAB_ITEMIDS["troponin"])].shape[0] > 0
    )
    row["orders_count_24h"] = _count_orders(prescriptions, intime, wend24)

    # CXR via chartevents (item 221214 = chest x-ray)
    cxr_items = [221214, 227680]
    cxr_sub = ce_24[ce_24["itemid"].isin(cxr_items)]
    row["has_cxr_24h"] = int(len(cxr_sub) > 0)
    row["cxr_count_24h"] = int(len(cxr_sub))

    # 24h 命名为历史兼容；短窗口严格截断在当前预测时点。
    wend24 = min(wend24, window_end)
    row["orders_count_24h"] = _count_orders(prescriptions, intime, wend24)

    # --- Mixed features (24h) ---
    row["vasopressors_24h"] = int(
        inputevents[
            (inputevents["starttime"] >= intime) &
            (inputevents["starttime"] < wend24) &
            (inputevents["itemid"].isin(VASOPRESSOR_ITEMIDS)) &
            (inputevents["storetime"] < window_end)
        ].shape[0] > 0
    )
    row["mechanical_ventilation_24h"] = int(
        procedureevents[
            (procedureevents["starttime"] >= intime) &
            (procedureevents["starttime"] < wend24) &
            (procedureevents["itemid"].isin(MV_ITEMIDS)) &
            (procedureevents["storetime"] < window_end)
        ].shape[0] > 0
    )
    presc_stay = prescriptions[
        (prescriptions["starttime"] >= intime) &
        (prescriptions["starttime"] < wend24) &
        (prescriptions["order_time"] < window_end)
    ]
    abx_mask = presc_stay["drug"].str.lower().str.contains(
        "|".join(ANTIBIOTIC_KEYWORDS), na=False
    )
    row["antibiotics_24h"] = int(abx_mask.any())
    row["blood_culture_sent_24h"] = row["has_blood_culture_24h"]

    # --- Labels ---
    row.update(_extract_labels(stay, hadm_id, labevents, inputevents, procedureevents,
                               chartevents, window_h, sofa_df))
    return row


def _count_orders(prescriptions: pd.DataFrame,
                  t_start: pd.Timestamp, t_end: pd.Timestamp) -> int:
    sub = prescriptions[
        (prescriptions["starttime"] >= t_start) &
        (prescriptions["starttime"] < t_end) &
        (prescriptions["order_time"] < t_end)
    ]
    return int(len(sub))


def _extract_labels(
    stay: pd.Series,
    hadm_id: int,
    labevents: pd.DataFrame,
    inputevents: pd.DataFrame,
    procedureevents: pd.DataFrame,
    chartevents: pd.DataFrame,
    window_h: int,
    sofa_df: dict,
) -> dict:
    stay_id = stay["stay_id"]
    intime = stay["intime"]
    outtime = stay["outtime"]
    window_end = intime + pd.Timedelta(hours=window_h)

    # ICU mortality

    # 精确住院死亡时间；日期级 DOD 不作为小时级时间。
    death_time = stay.get("deathtime", pd.NaT)
    expired = stay.get("hospital_expire_flag", 0) == 1
    if expired and (pd.isna(death_time) or death_time > stay.get("dischtime", pd.NaT)):
        icu_mort = np.nan
        inhosp_mort = np.nan
    else:
        icu_mort = int(expired and window_end < death_time <= outtime) if expired else 0
        inhosp_mort = int(expired and death_time > window_end) if expired else 0

    # AKI: creatinine rise >= 0.3 or >= 1.5x baseline within 48h after window
    aki = _compute_aki(labevents, intime, window_end, stay.get("dischtime"))

    # Sepsis deterioration: SOFA rise >= 2 within 24h after window
    sep_det = _compute_sepsis_deterioration(chartevents, window_end, outtime)

    # Escalation: vasopressors / MV / ICU transfer within 24h after window
    esc = _compute_escalation(inputevents, procedureevents, window_end, outtime)

    return {
        "icu_mortality": icu_mort,
        "inhosp_mortality": inhosp_mort,
        "aki": aki,
        "sepsis_deterioration": sep_det,
        "escalation": esc,
    }


def _compute_aki(labevents: pd.DataFrame,
                 intime: pd.Timestamp, window_end: pd.Timestamp,
                 discharge_time: pd.Timestamp) -> float:
    cr = labevents[
        labevents["itemid"].isin(LAB_ITEMIDS["creatinine"])
    ].sort_values("charttime")
    cr = cr[cr["valuenum"].notna()].sort_values(["available_time", "charttime"])
    if len(cr) < 2:
        return np.nan
    baseline = cr[cr["available_time"] < window_end]["valuenum"]
    if len(baseline) == 0:
        return np.nan
    baseline_val = float(baseline.iloc[-1])
    post = cr[
        (cr["charttime"] > window_end) &
        (cr["charttime"] <= window_end + pd.Timedelta(hours=48))
    ]["valuenum"]
    if len(post) == 0:
        return np.nan
    peak = float(post.max())
    if (peak - baseline_val >= 0.3) or (peak >= 1.5 * baseline_val):
        return 1.0
    return 0.0 if pd.notna(discharge_time) and discharge_time >= window_end + pd.Timedelta(hours=48) else np.nan


def _gcs_total(chartevents: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> float:
    GCS_EYE, GCS_VBL, GCS_MOT = 220739, 223900, 223901
    sub = chartevents[
        (chartevents["charttime"] >= t_start) &
        (chartevents["charttime"] < t_end) &
        (chartevents["itemid"].isin([GCS_EYE, GCS_VBL, GCS_MOT]))
    ]
    if sub.empty:
        return np.nan
    last = sub.sort_values("charttime").groupby("itemid")["valuenum"].last()
    if len(last) < 3:
        return np.nan
    return float(last.sum())


def _compute_sepsis_deterioration(
    chartevents: pd.DataFrame,
    window_end: pd.Timestamp,
    outtime: pd.Timestamp,
) -> float:
    pre_start = window_end - pd.Timedelta(hours=6)
    post_end = window_end + pd.Timedelta(hours=24)

    # GCS proxy: drop >= 3 points ≈ neuro SOFA +2
    gcs_pre = _gcs_total(chartevents, pre_start, window_end)
    gcs_post = _gcs_total(chartevents, window_end, post_end)
    if pd.notna(gcs_pre) and pd.notna(gcs_post) and (gcs_pre - gcs_post) >= 3:
        return 1

    # MAP proxy: MAP drops below 70 after window (hemodynamic deterioration)
    MAP_ITEMIDS = [220052, 220181, 225312]
    pre_map = chartevents[
        (chartevents["charttime"] >= pre_start) &
        (chartevents["available_time"] < window_end) &
        (chartevents["itemid"].isin(MAP_ITEMIDS))
    ]["valuenum"].dropna()
    post_map = chartevents[
        (chartevents["charttime"] >= window_end) &
        (chartevents["charttime"] < post_end) &
        (chartevents["itemid"].isin(MAP_ITEMIDS))
    ]["valuenum"].dropna()
    map_known = len(pre_map) > 0 and len(post_map) > 0
    if map_known and float(pre_map.median()) >= 70 and float(post_map.min()) < 65:
        return 1
    gcs_known = pd.notna(gcs_pre) and pd.notna(gcs_post)
    return 0.0 if map_known and gcs_known and outtime >= post_end else np.nan


def _compute_escalation(
    inputevents: pd.DataFrame, procedureevents: pd.DataFrame,
    window_end: pd.Timestamp, outtime: pd.Timestamp,
) -> float:
    """截点后 24 小时 ICU 内新启动升阶治疗；随访不足且未发生时记未知。"""
    t_end = window_end + pd.Timedelta(hours=24)
    vasopress = inputevents[
        (inputevents["starttime"] > window_end) &
        (inputevents["starttime"] <= t_end) &
        (inputevents["itemid"].isin(VASOPRESSOR_ITEMIDS))
    ]
    mv = procedureevents[
        (procedureevents["starttime"] > window_end) &
        (procedureevents["starttime"] <= t_end) &
        (procedureevents["itemid"].isin(MV_ITEMIDS))
    ]
    if len(vasopress) or len(mv):
        return 1.0
    return 0.0 if outtime >= t_end else np.nan


def _compute_comorbidities(diagnoses: pd.DataFrame) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for hadm_id, grp in diagnoses.groupby("hadm_id"):
        codes = grp["icd_code"].astype(str).tolist()
        rec: dict = {}
        for key, prefixes in COMORBIDITY_MAP.items():
            rec[key] = int(any(c.startswith(p) for c in codes for p in prefixes))
        out[hadm_id] = rec
    return out


def _compute_dx_categories(diagnoses: pd.DataFrame) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for hadm_id, grp in diagnoses.groupby("hadm_id"):
        codes = grp["icd_code"].astype(str).tolist()
        rec: dict = {}
        matched = False
        for key, prefixes in DX_MAP.items():
            hit = int(any(c.startswith(p) for c in codes for p in prefixes))
            rec[key] = hit
            if hit:
                matched = True
        rec["dx_other"] = int(not matched)
        out[hadm_id] = rec
    return out



def _encode_admission_source(loc: str) -> int:
    loc = str(loc).upper()
    if "EMERGENCY" in loc:
        return 1
    if "TRANSFER" in loc or "HOSPITAL" in loc:
        return 2
    if "ELECTIVE" in loc or "PROCEDURE" in loc:
        return 3
    return 0
