"""流式抽取索引 ICU 成人住院的 POE 录入时间，供处方输入审计。"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/poe_order_times.parquet"))
    args = parser.parse_args()
    raw = Path(yaml.safe_load(args.config.read_text())["data"]["mimic_iv_dir"])
    stays = pd.read_csv(raw / "icu/icustays.csv.gz", usecols=["subject_id", "hadm_id", "intime"])
    stays = stays.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id")
    patients = pd.read_csv(raw / "hosp/patients.csv.gz", usecols=["subject_id", "anchor_age"])
    stays = stays.merge(patients, on="subject_id", validate="many_to_one")
    hadms = set(stays.loc[stays.anchor_age.ge(18), "hadm_id"])
    chunks = []
    for number, chunk in enumerate(pd.read_csv(raw / "hosp/poe.csv.gz",
                                                usecols=["hadm_id", "poe_id", "ordertime"],
                                                chunksize=250_000, low_memory=False)):
        selected = chunk.loc[chunk.hadm_id.isin(hadms) & chunk.poe_id.notna() & chunk.ordertime.notna(),
                             ["poe_id", "ordertime"]]
        if len(selected):
            chunks.append(selected)
        if number % 50 == 0:
            print("poe chunks", number, "selected", sum(len(c) for c in chunks), flush=True)
    result = pd.concat(chunks, ignore_index=True)
    result["ordertime"] = pd.to_datetime(result.ordertime, errors="coerce")
    result = result.dropna().groupby("poe_id", sort=False, as_index=False).ordertime.min()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    result.to_parquet(temp, index=False)
    temp.replace(args.output)
    print("saved POE order times", len(result), args.output, flush=True)

if __name__ == "__main__":
    main()
