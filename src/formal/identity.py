"""模型、预测和评价身份；只散列真正参与该阶段的内容。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    """按患者顺序、列序和值散列；与旁列及 parquet 编码无关。"""
    x = frame.loc[:, columns].copy()
    h = hashlib.sha256()
    h.update(canonical_hash(columns).encode())
    h.update(pd.util.hash_pandas_object(x, index=False, categorize=False).values.tobytes())
    return h.hexdigest()


def fit_identity(train: pd.DataFrame, validation: pd.DataFrame, label: str,
                 features: list[str], model: str, seed: int, model_config: dict,
                 preprocessor_version: str, implementation_version: str,
                 validation_used: bool) -> str:
    cols = ["subject_id", "stay_id", label, *features]
    return canonical_hash({"train": frame_hash(train, cols),
                           "validation": frame_hash(validation, cols) if validation_used else None,
                           "features": features, "model": model, "seed": seed,
                           "model_config": model_config,
                           "preprocessor_version": preprocessor_version,
                           "implementation_version": implementation_version})


def prediction_identity(fit_id: str, frame: pd.DataFrame, features: list[str]) -> str:
    return canonical_hash({"fit_id": fit_id,
                           "inputs": frame_hash(frame, ["subject_id", "stay_id", *features])})


def evaluation_identity(frame: pd.DataFrame, label: str, cohort_id: str,
                        endpoint_version: str) -> str:
    return canonical_hash({"cohort_id": cohort_id, "endpoint_version": endpoint_version,
                           "labels": frame_hash(frame, ["subject_id", "stay_id", label])})
