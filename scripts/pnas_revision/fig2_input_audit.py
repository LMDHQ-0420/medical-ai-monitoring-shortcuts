"""核对正式格点的实际特征和预处理器，输出 Fig. 2 的重跑范围。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.pnas_revision.run_tabular import MODELS, SEEDS, TASKS, WINDOWS, digest_file


LATE = {
    "sofa_score", "has_diabetes", "has_ckd", "has_heart_failure",
    "has_copd", "has_cancer", "dx_infection", "dx_cardiovascular",
    "dx_respiratory", "dx_renal", "dx_other",
}


def audit(run: Path) -> None:
    rows = []
    data_root = Path(__file__).resolve().parents[2] / "data/processed"
    hashes = {w: digest_file(data_root / f"mimic_iv_{w}h.parquet") for w in WINDOWS}
    protocol_hash = digest_file(run / "analysis_protocol.json")
    for task in TASKS:
        for model in MODELS:
            for window in WINDOWS:
                for seed in SEEDS:
                    cell = run / "formal/tabular" / task / model / f"seed{seed}" / f"{window}h"
                    manifest_path = cell / "manifest.json"
                    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                    for group in ("D", "DT", "F"):
                        transformer_path = cell / f"transformer_{group}.joblib"
                        transformer = joblib.load(transformer_path) if transformer_path.exists() else None
                        declared = manifest.get("feature_columns", {}).get(group, [])
                        actual = list(transformer.feature_cols) if transformer is not None else []
                        used = sorted(LATE.intersection(actual))
                        valid = (
                            manifest.get("status") == "complete"
                            and manifest.get("run_type") == "formal"
                            and manifest.get("data_sha256") == hashes[window]
                            and manifest.get("protocol_sha256") == protocol_hash
                            and (cell / "predictions.csv.gz").exists()
                            and digest_file(cell / "predictions.csv.gz") == manifest.get("predictions_sha256")
                            and declared == actual
                            and transformer is not None
                            and list(transformer._medians.index) == actual
                            and not used
                            and (cell / f"model_{group}.joblib").exists()
                            and (cell / "predictions.csv.gz").exists()
                        )
                        rows.append({
                            "task": task, "model": model, "window_h": window,
                            "seed": seed, "group": group, "status": "qualified" if valid else "rerun_required",
                            "late_fields_used": "|".join(used), "n_features": len(actual),
                            "manifest_matches_transformer": declared == actual,
                            "data_hash_matches_current": manifest.get("data_sha256") == hashes[window],
                            "manifest_path": str(manifest_path),
                            "transformer_path": str(transformer_path),
                            "reason": "" if valid else "late_field_or_missing_or_mismatched_artifact",
                        })
    sequence_hash = digest_file(data_root / "mimic_iv_hourly.npz")
    for task in TASKS:
        for window in WINDOWS:
            for seed in SEEDS:
                cell = run / "formal/gru_seq" / task / f"seed{seed}" / f"{window}h"
                path = cell / "manifest.json"
                m = json.loads(path.read_text()) if path.exists() else {}
                used = sorted(LATE.intersection(m.get("static_D", [])))
                valid = (m.get("status") == "complete" and m.get("static_D") == ["age", "gender"]
                         and m.get("sequence_sha256") == sequence_hash
                         and m.get("data_sha256") == hashes[window] and not used
                         and (cell / "predictions.csv.gz").exists()
                         and digest_file(cell / "predictions.csv.gz") == m.get("predictions_sha256"))
                for group in ("D", "DT", "F"):
                    model_path = cell / f"model_{group}.pt"
                    actual_dims = torch.load(model_path, map_location="cpu", weights_only=True) if model_path.exists() else {}
                    expected_input = {"D": 14, "DT": 17, "F": 59}[group]
                    dims_valid = actual_dims.get("input_size") == expected_input and actual_dims.get("static_size") == 2
                    rows.append({
                        "task": task, "model": "gru_seq", "window_h": window,
                        "seed": seed, "group": group, "status": "qualified" if valid and dims_valid else "rerun_required",
                        "late_fields_used": "|".join(used), "n_features": actual_dims.get("input_size", ""),
                        "manifest_matches_transformer": True,
                        "data_hash_matches_current": m.get("data_sha256") == hashes[window],
                        "manifest_path": str(path), "transformer_path": "static_D recorded in manifest",
                        "reason": "" if valid else "GRU_static_branch_unqualified",
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(run / "fig2_field_impact_scope.csv", index=False)
    summary = {
        "tabular_group_cells": 625 * 3,
        "gru_group_cells": 125 * 3,
        "qualified": int(frame.status.eq("qualified").sum()),
        "rerun_required": int(frame.status.eq("rerun_required").sum()),
        "late_fields_checked": sorted(LATE),
        "conclusion": "仅合格格点可冻结用于 Fig. 2 测试集重评估",
    }
    (run / "fig2_field_impact_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    audit(parser.parse_args().run_dir.resolve())
