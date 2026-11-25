"""核验 Fig. 2 的截点、模型输入、匹配对子、预测完整性及图件。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(run: Path) -> None:
    impact = pd.read_csv(run / "fig2_field_impact_scope.csv")
    assert len(impact) == 2250 and impact.status.eq("qualified").all()
    state = pd.read_parquet(ROOT / "data/processed/mimic_iv_fig2_clinical_state.parquet")
    assert len(state) == 245100 and not state.duplicated(["stay_id", "window_h"]).any()
    future_events = 0
    for vital in ("hr", "sbp", "dbp", "temp", "rr", "spo2"):
        observed = state[vital + "_observed"]
        future_events += int((state.loc[observed, vital + "_available_time"] >=
                              state.loc[observed, "prediction_cutoff"]).sum())
        future_events += int((state.loc[observed, vital + "_charttime"] >
                              state.loc[observed, vital + "_available_time"]).sum())
    assert future_events == 0
    protocol = json.loads((run / "fig2_matching/matching_protocol.json").read_text())
    assert len(protocol) == 25 and all(x["balance_pass"] for x in protocol)
    pairs = pd.read_csv(run / "fig2_matching/matched_pairs.csv.gz")
    assert not pairs.duplicated(["window_h", "seed", "high_stay_id"]).any()
    assert not pairs.duplicated(["window_h", "seed", "low_stay_id"]).any()
    for (_, _), group in pairs.groupby(["window_h", "seed"]):
        assert not set(group.high_stay_id).intersection(group.low_stay_id)
    metrics = pd.read_csv(run / "matched_seed_metrics.csv.gz")
    assert len(metrics) == 625
    files = 0
    for row in metrics.itertuples(index=False):
        path = ROOT / row.prediction_file
        assert path.exists() and sha(path) == row.prediction_sha256
        prediction = pd.read_csv(path)
        assert len(prediction) == row.n_matched == row.n_pairs * 2
        assert not prediction.stay_id.duplicated().any()
        assert prediction.groupby("pair_index").size().eq(2).all()
        assert int(prediction.y_true.sum()) == row.positive_matched
        files += 1
    diagnosis = pd.read_parquet(ROOT / "data/processed/eicu_fig2_diagnosis.parquet")
    assert len(diagnosis) == 652080
    assert not (diagnosis.last_entry_offset_min >= diagnosis.window_h * 60).any()
    assert len(pd.read_csv(run / "context_seed_metrics.csv.gz")) == 2500
    assert len(pd.read_csv(run / "diagnosis_seed_metrics.csv.gz")) == 3750
    assert len(pd.read_csv(run / "matched_refit_sensitivity.csv.gz")) == 25
    for name in ("fig2", "extended_eicu_diagnosis"):
        assert (run / "figures" / f"{name}.pdf").exists()
        assert (run / "figures" / f"{name}.png").exists()
    report = {
        "status": "passed", "qualified_model_groups": len(impact),
        "clinical_state_rows": len(state), "future_time_violations": future_events,
        "balanced_test_splits": len(protocol), "matching_pairs_total": len(pairs),
        "matched_seed_cells": len(metrics), "paired_prediction_files_sha256_checked": files,
        "eicu_diagnosis_rows": len(diagnosis), "eicu_after_cutoff_violations": 0,
        "context_rows": 2500, "eicu_stratum_rows": 3750, "refit_sensitivity_cells": 25,
    }
    (run / "fig2_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    validate(parser.parse_args().run_dir.resolve())
