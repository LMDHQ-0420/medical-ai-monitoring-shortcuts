"""正式表格模型训练与患者级推理；复用依据是实际拟合和预测身份。"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data.feature_groups import PREDICTOR_SETS, features_for
from src.data.transforms import EHRTransformer
from src.formal.identity import canonical_hash, evaluation_identity, file_hash, fit_identity, prediction_identity
from src.formal.splits import split_frame, split_id
from src.models.train_utils import create_and_fit

MODELS = ("lr", "rf", "xgb", "lgbm", "mlp")
TASKS = ("icu_mortality", "inhosp_mortality", "aki", "sepsis_deterioration", "escalation")
WINDOWS = (3, 6, 12, 18, 24)
SEEDS = (42, 43, 44, 45, 46)
MODEL_CONFIG_KEY = {"lr": "logistic", "rf": "random_forest", "xgb": "xgboost",
                    "lgbm": "lightgbm", "mlp": "mlp"}
IMPLEMENTATION_VERSION = "landmark-tabular-v2"
PREPROCESSOR_VERSION = "median-zscore-train-only-v1"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(temp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.csv.gz")
    frame.to_csv(temp, index=False, compression="gzip", float_format="%.17g")
    os.replace(temp, path)


def group_dir(output_root: Path, task: str, model: str, window_h: int,
              seed: int, state_representation: str, predictor_set: str) -> Path:
    return (output_root / "models/tabular" / task / model / f"seed{seed}" /
            f"{window_h}h" / state_representation / predictor_set)


def prepare_frame(data_root: Path, task: str, window_h: int,
                  state_representation: str) -> pd.DataFrame:
    frame = pd.read_parquet(data_root / f"mimic_iv_{window_h}h.parquet")
    if state_representation == "first_plus_latest":
        if window_h != 24 or task != "inhosp_mortality":
            raise ValueError("状态增强仅用于 24 h Hospital Death")
        side = pd.read_parquet(data_root / "mimic_iv_clinical_state.parquet")
        side = side.loc[side.window_h.eq(window_h)].copy()
        if side.duplicated(["stay_id", "window_h"]).any():
            raise ValueError("最新生命体征侧表主键重复")
        selected = side[["stay_id", "window_h", *[f"{v}_value" for v in ("hr", "sbp", "dbp", "temp", "rr", "spo2")]]]
        frame = frame.merge(selected, on=["stay_id", "window_h"], how="left", validate="one_to_one")
        frame = frame.rename(columns={f"{v}_value": f"{v}_latest" for v in ("hr", "sbp", "dbp", "temp", "rr", "spo2")})
    return frame.loc[frame[task].notna()].copy()


def fit_group(config: dict, data_root: Path, output_root: Path, protocol: dict,
              task: str, model: str, window_h: int, seed: int,
              state_representation: str, predictor_set: str,
              fixed_split: pd.DataFrame, model_artifact_root: Path | None = None,
              *, action: str = "train") -> dict:
    if model not in MODELS or task not in TASKS or window_h not in WINDOWS or seed not in SEEDS:
        raise ValueError("配置不属于正式网格")
    features = features_for(predictor_set, state_representation)
    frame = prepare_frame(data_root, task, window_h, state_representation)
    train, validation, test = split_frame(frame, fixed_split, seed)
    if train[task].nunique() < 2 or validation[task].nunique() < 2 or test[task].nunique() < 2:
        raise ValueError("划分后的结局缺少类别")
    cfg = copy.deepcopy(config)
    cfg["training"]["seed"] = seed
    cfg["training"]["random_forest"]["n_jobs"] = 1
    cfg["training"].setdefault("xgboost", {})["n_jobs"] = 1
    cfg["training"].setdefault("lightgbm", {})["n_jobs"] = 1
    settings = cfg["training"][MODEL_CONFIG_KEY[model]]
    validation_used = model in ("xgb", "lgbm")
    fit_id = fit_identity(train, validation, task, features, model, seed, settings,
                          PREPROCESSOR_VERSION, IMPLEMENTATION_VERSION, validation_used)
    pred_id = prediction_identity(fit_id, test, features)
    cohort_id = canonical_hash({"dataset": "mimic_iv", "window_h": window_h,
                                "stays": sorted(frame.stay_id.astype(str).tolist())})
    eval_id = evaluation_identity(test, task, cohort_id, protocol["endpoint_version"])
    cell = group_dir(model_artifact_root or output_root, task, model, window_h,
                     seed, state_representation, predictor_set)
    manifest_path = cell / "manifest.json"
    model_path, transformer_path = cell / "model.joblib", cell / "transformer.joblib"
    pred_path = cell / "internal_predictions.csv.gz"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    fit_valid = (previous.get("status") == "complete" and previous.get("fit_id") == fit_id
                 and model_path.exists() and transformer_path.exists()
                 and previous.get("model_sha256") == file_hash(model_path)
                 and previous.get("transformer_sha256") == file_hash(transformer_path))
    probability_valid = (fit_valid and previous.get("prediction_id") == pred_id
                         and pred_path.exists() and previous.get("prediction_sha256") == file_hash(pred_path))
    prediction_valid = probability_valid and previous.get("evaluation_id") == eval_id
    if action == "plan":
        return {"fit_id": fit_id, "evaluation_id": eval_id,
                "fit_action": "reuse" if fit_valid else "retrain",
                "prediction_action": "reuse" if prediction_valid else "relabel" if probability_valid else "infer",
                "cell": str(cell), "n_train": len(train), "n_validation": len(validation), "n_test": len(test)}
    if action == "infer" and not fit_valid:
        raise ValueError("冻结模型无有效拟合身份")
    if action not in ("train", "infer"):
        raise ValueError(action)
    cell.mkdir(parents=True, exist_ok=True)
    if not fit_valid:
        transformer = EHRTransformer(features)
        x_train = transformer.fit_transform(train)
        x_val = transformer.transform(validation)
        fitted = create_and_fit(model, cfg, x_train, train[task].astype(int).to_numpy(),
                                x_val, validation[task].astype(int).to_numpy())
        temp_model = cell / "model.joblib.tmp"
        temp_transformer = cell / "transformer.joblib.tmp"
        joblib.dump(fitted, temp_model)
        joblib.dump(transformer, temp_transformer)
        os.replace(temp_model, model_path)
        os.replace(temp_transformer, transformer_path)
        fit_source = "new_fit"
    else:
        fit_source = previous.get("fit_source", "verified_existing_fit")
        fitted = joblib.load(model_path)
        transformer = joblib.load(transformer_path)
    if not prediction_valid:
        if probability_valid:
            saved = pd.read_csv(pred_path)
            if not np.array_equal(saved.subject_id.to_numpy(), test.subject_id.to_numpy()) or not np.array_equal(saved.stay_id.to_numpy(), test.stay_id.to_numpy()):
                raise ValueError("复用概率与当前测试患者顺序不一致")
            scores = saved.probability.to_numpy()
        else:
            scores = np.asarray(fitted.predict_proba(transformer.transform(test))).reshape(-1)
        if len(scores) != len(test) or not np.isfinite(scores).all():
            raise ValueError("预测概率缺失或非有限")
        prediction = pd.DataFrame({
            "dataset": "mimic_iv", "subject_id": test.subject_id.to_numpy(),
            "stay_id": test.stay_id.to_numpy(), "hospital_id": np.nan,
            "age": test.age.to_numpy(), "task": task, "model": model,
            "window_h": window_h, "seed": seed,
            "state_representation": state_representation, "predictor_set": predictor_set,
            "y_true": test[task].astype(int).to_numpy(), "probability": scores,
            "fit_id": fit_id, "prediction_id": pred_id, "evaluation_id": eval_id,
        })
        atomic_csv(pred_path, prediction)
    manifest = {
        "status": "complete", "task": task, "model": model, "window_h": window_h,
        "seed": seed, "state_representation": state_representation,
        "predictor_set": predictor_set, "features": features,
        "data_root": str(data_root.resolve()), "output_root": str(output_root.resolve()),
        "model_artifact_root": str((model_artifact_root or output_root).resolve()),
        "fit_id": fit_id, "prediction_id": pred_id, "evaluation_id": eval_id,
        "cohort_id": cohort_id, "split_id": split_id(fixed_split, seed),
        "endpoint_version": protocol["endpoint_version"],
        "analysis_id": protocol["analysis_id"], "fit_source": fit_source,
        "validation_used": validation_used, "model_config": settings,
        "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
        "n_positive_test": int(test[task].sum()),
        "model_sha256": file_hash(model_path),
        "transformer_sha256": file_hash(transformer_path),
        "prediction_sha256": file_hash(pred_path),
    }
    atomic_json(manifest_path, manifest)
    return manifest
