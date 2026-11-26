"""从截点前生命体征事件提取最后可用值及精确测量、可用时间。"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (3, 6, 12, 18, 24)
VITAL_IDS = {
    "hr": (220045,), "sbp": (220179, 220050),
    "dbp": (220180, 220051), "temp": (223761, 223762, 226329),
    "rr": (220210, 224690), "spo2": (220277,),
}
ID_TO_NAME = {item: name for name, ids in VITAL_IDS.items() for item in ids}


def extract_one(arg: tuple[int, pd.Timestamp]) -> list[dict]:
    stay_id, intime = arg
    path = ROOT / "data/processed/cache/chartevents" / f"{stay_id}.parquet"
    events = pd.DataFrame()
    if path.exists():
        table = pq.read_table(path, columns=["itemid", "charttime", "storetime", "valuenum"])
        table = table.filter(pc.is_in(table["itemid"], value_set=pa.array(list(ID_TO_NAME))))
        events = table.to_pandas()
    grouped = {}
    if not events.empty:
        events["charttime"] = pd.to_datetime(events.charttime, errors="coerce", format="mixed")
        events["storetime"] = pd.to_datetime(events.storetime, errors="coerce", format="mixed")
        events["available_time"] = events[["charttime", "storetime"]].max(axis=1)
        events["valuenum"] = pd.to_numeric(events.valuenum, errors="coerce")
        events = events.loc[
            events.charttime.ge(intime) & events.available_time.ge(intime)
            & events.available_time.lt(intime + pd.Timedelta(hours=24))
            & events.valuenum.notna()
        ].copy()
        events["name"] = events.itemid.map(ID_TO_NAME)
        events = events.sort_values(["available_time", "charttime"])
        for name, group in events.groupby("name"):
            grouped[name] = (group.available_time.to_numpy(dtype="datetime64[ns]"),
                             group.charttime.to_numpy(dtype="datetime64[ns]"),
                             group.valuenum.to_numpy(dtype=float), group.itemid.to_numpy(dtype=int))
    rows = []
    for window in WINDOWS:
        cutoff = intime + pd.Timedelta(hours=window)
        row = {"stay_id": int(stay_id), "window_h": window, "prediction_cutoff": cutoff}
        for name in VITAL_IDS:
            group = grouped.get(name)
            pos = np.searchsorted(group[0], np.datetime64(cutoff), side="left") - 1 if group is not None else -1
            present = pos >= 0
            row[name + "_observed"] = bool(present)
            row[name + "_value"] = float(group[2][pos]) if present else np.nan
            row[name + "_charttime"] = pd.Timestamp(group[1][pos]) if present else pd.NaT
            row[name + "_available_time"] = pd.Timestamp(group[0][pos]) if present else pd.NaT
            if present and name == "temp":
                value = row[name + "_value"]
                if int(group[3][pos]) == 223761 or 80 <= value <= 115:
                    value = (value - 32) * 5 / 9
                row[name + "_value"] = value if 25 <= value <= 45 else np.nan
                row[name + "_observed"] = bool(np.isfinite(row[name + "_value"]))
        rows.append(row)
    return rows


def build() -> None:
    cohort = pd.read_parquet(ROOT / "data/processed/mimic_iv_24h.parquet", columns=["stay_id"])
    stays = pd.read_csv(
        ROOT / "data/raw/mimic_iv/mimic-iv-2.2/icu/icustays.csv.gz",
        usecols=["stay_id", "intime"],
    )
    stays.intime = pd.to_datetime(stays.intime, errors="coerce", format="mixed")
    frame = cohort.merge(stays, on="stay_id", validate="one_to_one")
    if frame.intime.isna().any():
        raise ValueError("ICU 入科时间缺失")
    out = ROOT / "data/processed/mimic_iv_fig2_clinical_state.parquet"
    temp = out.with_suffix(".tmp.parquet")
    writer = None
    inputs = list(frame[["stay_id", "intime"]].itertuples(index=False, name=None))
    with ThreadPoolExecutor(max_workers=8) as pool:
        batch = []
        for i, rows in enumerate(pool.map(extract_one, inputs), 1):
            batch.extend(rows)
            if i % 1000 == 0 or i == len(inputs):
                table = pa.Table.from_pandas(pd.DataFrame(batch), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table)
                batch.clear()
                if i % 5000 == 0:
                    print("stays processed", i, flush=True)
    if writer is None:
        raise ValueError("没有生命体征数据")
    writer.close()
    temp.replace(out)
    result = pd.read_parquet(out)
    if len(result) != len(inputs) * len(WINDOWS):
        raise ValueError("提取结果行数不符")
    report = {
        "source": "data/processed/cache/chartevents/{stay_id}.parquet",
        "source_event_time": "charttime",
        "availability_time": "max(charttime,storetime)",
        "selection": "[ICU intime, prediction cutoff); latest availability time, then latest charttime",
        "rows": len(result),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "coverage": result.groupby("window_h")[[v + "_observed" for v in VITAL_IDS]].mean().to_dict("index"),
    }
    (ROOT / "data/processed/mimic_iv_fig2_clinical_state_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print("complete", len(result), out, flush=True)
