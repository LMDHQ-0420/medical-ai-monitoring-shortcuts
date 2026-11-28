"""eICU 冻结模型推理；仅在输入或权重变化时重新计算患者概率。"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data.feature_groups import features_for
from src.exp1.trainer import atomic_csv, atomic_json, group_dir
from src.formal.identity import canonical_hash, evaluation_identity, file_hash, prediction_identity


def infer_group(data_root: Path, output_root: Path, protocol: dict,
                task: str, model: str, window_h: int, seed: int,
                predictor_set: str) -> dict:
    cell = group_dir(output_root, task, model, window_h, seed,
                     "first_measurements", predictor_set)
    internal_manifest = json.loads((cell / "manifest.json").read_text())
    model_path, transformer_path = cell / "model.joblib", cell / "transformer.joblib"
    if (internal_manifest["status"] != "complete" or
        file_hash(model_path) != internal_manifest["model_sha256"] or
        file_hash(transformer_path) != internal_manifest["transformer_sha256"]):
        raise ValueError(f"冻结权重来源校验失败: {cell}")
    frame = pd.read_parquet(data_root / f"eicu_{window_h}h.parquet")
    frame = frame.loc[frame[task].notna()].copy()
    features = features_for(predictor_set)
    pred_id = prediction_identity(internal_manifest["fit_id"], frame, features)
    cohort_id = canonical_hash({"dataset": "eicu", "window_h": window_h,
                                "stays": sorted(frame.stay_id.astype(str).tolist())})
    eval_id = evaluation_identity(frame, task, cohort_id, protocol["endpoint_version"])
    pred_path, manifest_path = cell / "external_predictions.csv.gz", cell / "external_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    probability_valid = (previous.get("status") == "complete" and previous.get("prediction_id") == pred_id
                         and pred_path.exists() and previous.get("prediction_sha256") == file_hash(pred_path))
    eval_valid = probability_valid and previous.get("evaluation_id") == eval_id
    if eval_valid:
        return previous
    if probability_valid:
        saved = pd.read_csv(pred_path)
        if not np.array_equal(saved.subject_id.astype(str), frame.subject_id.astype(str)) or not np.array_equal(saved.stay_id, frame.stay_id):
            raise ValueError("外部概率与患者顺序不符")
        scores = saved.probability.to_numpy()
        source = "relabel_existing_probability"
    else:
        transformer = joblib.load(transformer_path)
        fitted = joblib.load(model_path)
        scores = np.asarray(fitted.predict_proba(transformer.transform(frame))).reshape(-1)
        source = "frozen_model_inference"
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("外部预测概率不完整")
    prediction = pd.DataFrame({"dataset": "eicu", "subject_id": frame.subject_id.to_numpy(),
                               "stay_id": frame.stay_id.to_numpy(), "hospital_id": frame.hospital_id.to_numpy(),
                               "age": frame.age.to_numpy(), "task": task, "model": model,
                               "window_h": window_h, "seed": seed,
                               "state_representation": "first_measurements", "predictor_set": predictor_set,
                               "y_true": frame[task].astype(int).to_numpy(), "probability": scores,
                               "fit_id": internal_manifest["fit_id"], "prediction_id": pred_id,
                               "evaluation_id": eval_id})
    atomic_csv(pred_path, prediction)
    manifest = {"status": "complete", "task": task, "model": model, "window_h": window_h,
                "seed": seed, "predictor_set": predictor_set,
                "fit_id": internal_manifest["fit_id"], "prediction_id": pred_id,
                "evaluation_id": eval_id, "cohort_id": cohort_id,
                "endpoint_version": protocol["endpoint_version"],
                "source": source, "n": len(frame), "n_positive": int(frame[task].sum()),
                "prediction_sha256": file_hash(pred_path)}
    atomic_json(manifest_path, manifest)
    return manifest


def read_external(root: Path, task: str, model: str, window_h: int, seed: int,
                  predictor_set: str):
    cell = group_dir(root, task, model, window_h, seed, "first_measurements", predictor_set)
    manifest = json.loads((cell / "external_manifest.json").read_text())
    path = cell / "external_predictions.csv.gz"
    if manifest["status"] != "complete" or file_hash(path) != manifest["prediction_sha256"]:
        raise ValueError(f"外部预测 SHA 校验失败: {path}")
    frame = pd.read_csv(path)
    if not frame.fit_id.eq(manifest["fit_id"]).all() or not frame.evaluation_id.eq(manifest["evaluation_id"]).all():
        raise ValueError("外部概率来源字段不一致")
    return manifest, frame
