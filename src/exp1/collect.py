"""只读取冻结模型输出，生成主分析正式表。"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.feature_groups import PREDICTOR_SETS
from src.exp1.trainer import MODELS, TASKS, WINDOWS, SEEDS, atomic_csv, group_dir
from src.formal.identity import file_hash
from src.formal.statistics import assess, gains, paired_interval


def read_group(root: Path, task: str, model: str, window_h: int,
               seed: int, state_representation: str, predictor_set: str):
    cell = group_dir(root, task, model, window_h, seed, state_representation, predictor_set)
    manifest = json.loads((cell / "manifest.json").read_text())
    path = cell / "internal_predictions.csv.gz"
    if manifest["status"] != "complete" or file_hash(path) != manifest["prediction_sha256"]:
        raise ValueError(f"患者预测来源校验失败: {cell}")
    frame = pd.read_csv(path)
    for key, value in (("fit_id", manifest["fit_id"]), ("prediction_id", manifest["prediction_id"]),
                       ("evaluation_id", manifest["evaluation_id"])):
        if not frame[key].eq(value).all():
            raise ValueError(f"{key} 与来源不一致: {cell}")
    return manifest, frame


def collect(root: Path, protocol: dict, n_bootstrap: int | None = None):
    count = protocol["n_bootstrap"] if n_bootstrap is None else n_bootstrap
    model_rows, gain_rows = [], []
    for task, model, window_h, seed in itertools.product(TASKS, MODELS, WINDOWS, SEEDS):
        frames, manifests, metrics = {}, {}, {}
        for predictor_set in PREDICTOR_SETS:
            manifest, frame = read_group(root, task, model, window_h, seed,
                                         "first_measurements", predictor_set)
            manifests[predictor_set], frames[predictor_set] = manifest, frame
            met = assess(frame.y_true.to_numpy(), frame.probability.to_numpy())
            metrics[predictor_set] = met
            model_rows.append({
                "analysis_id": protocol["analysis_id"], "task": task,
                "model": model, "window_h": window_h, "seed": seed,
                "state_representation": "first_measurements", "predictor_set": predictor_set,
                "cohort_id": manifest["cohort_id"], "split_id": manifest["split_id"],
                "endpoint_version": manifest["endpoint_version"],
                "fit_id": manifest["fit_id"], "evaluation_id": manifest["evaluation_id"],
                "fit_source": manifest["fit_source"], **met,
            })
        baseline = frames["clinical_state"]
        for name, frame in frames.items():
            if not np.array_equal(frame.subject_id, baseline.subject_id) or not np.array_equal(frame.stay_id, baseline.stay_id) or not np.array_equal(frame.y_true, baseline.y_true):
                raise ValueError(f"输入组患者或标签不一致: {task}/{model}/{window_h}/{seed}/{name}")
        g = gains(metrics)
        probability = {name: frame.probability.to_numpy() for name, frame in frames.items()}
        interval = paired_interval(baseline.y_true.to_numpy(), probability, count,
                                   f"exp1/{task}/{model}/{window_h}/{seed}")
        m = manifests["full_model"]
        gain_rows.append({"analysis_id": protocol["analysis_id"], "task": task,
                          "model": model, "window_h": window_h, "seed": seed,
                          "state_representation": "first_measurements", "cohort_id": m["cohort_id"],
                          "split_id": m["split_id"], "endpoint_version": m["endpoint_version"],
                          "evaluation_id": m["evaluation_id"],
                          "n": len(baseline), "n_positive": int(baseline.y_true.sum()),
                          "n_negative": int((1 - baseline.y_true).sum()),
                          "evaluable": metrics["full_model"]["evaluable"],
                          "unevaluable_reason": metrics["full_model"]["unevaluable_reason"],
                          **{f"fit_id_{k}": v["fit_id"] for k, v in manifests.items()},
                          **g, **interval})
    dest = root / "exp1"
    atomic_csv(dest / "model_metrics.csv.gz", pd.DataFrame(model_rows))
    atomic_csv(dest / "gain_metrics.csv.gz", pd.DataFrame(gain_rows))
    return len(model_rows), len(gain_rows)
