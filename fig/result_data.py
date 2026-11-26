"""从 outputs/exp1–exp5 读取正式结果，供原有面板样式使用。"""

from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent / "outputs"
LABEL_STANDARD = json.loads((ROOT / "label_standard.json").read_text(encoding="utf-8"))
OUTCOME_LABELS = {item["key"]: item["figure"] for item in LABEL_STANDARD["outcomes"]}
FIGURE_TERMS = {item["id"]: item["figure"] for section in
                ("metrics", "populations", "axes_and_strata", "contrasts")
                for item in LABEL_STANDARD[section]}
MODEL_CONFIGURATION_LINES = {item["id"]: "\n".join(item["figure_lines"])
                             for item in LABEL_STANDARD["model_configurations"]}
TASKS = ["icu_mortality", "inhosp_mortality", "aki", "sepsis_deterioration", "escalation"]
TASK_LABELS = [OUTCOME_LABELS[task] for task in TASKS]
MODEL_KEYS = ["lr", "rf", "xgb", "lgbm", "mlp"]
MODEL_LABELS = [next(item["figure"] for item in LABEL_STANDARD["models"]["main"]
                     if item["key"] == model) for model in MODEL_KEYS]
WINDOWS = [3, 6, 12, 18, 24]
SEEDS = [42, 43, 44, 45, 46]
AGES = ["<45", "45-64", "65-80", ">80"]
AGE_LABELS = ["<45", "45–64", "65–80", ">80"]
KEYS = ["task", "model", "window"]


def _read(experiment, name="results.csv.gz"):
    path = OUTPUTS / experiment / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少正式绘图输入：{path}")
    return pd.read_csv(path)


def _grid(frame, name):
    keys = ["task", "model", "window_h", "seed"]
    expected = pd.MultiIndex.from_product(
        [TASKS, MODEL_KEYS, WINDOWS, SEEDS], names=keys
    )
    actual = pd.MultiIndex.from_frame(frame[keys])
    if len(frame) != len(expected) or not actual.is_unique or not actual.sort_values().equals(expected.sort_values()):
        raise ValueError(f"{name} 不是完整的 625 格点五模型正式结果")
    return frame


@lru_cache(None)
def exp1():
    d = _grid(_read("exp1"), "exp1")
    return d.rename(columns={"window_h": "window", "auc_D": "disease",
                             "auc_O": "observation", "auc_F": "mixed",
                             "odi_raw": "odi"})


@lru_cache(None)
def exp2():
    matched = _grid(_read("exp2"), "exp2")
    context = _read("exp2", "context.csv.gz")
    return matched.rename(columns={"window_h": "window"}), context.rename(columns={"window_h": "window"})


@lru_cache(None)
def balance():
    return _read("exp2", "balance.csv")


@lru_cache(None)
def exp3():
    d = _grid(_read("exp3"), "exp3")
    return d.rename(columns={"window_h": "window", "odi_raw": "original",
                             "odi_residual": "residual", "reference_gain": "mixed_change"})


@lru_cache(None)
def age():
    d = _read("exp4")
    return d.rename(columns={"window_h": "window", "dataset": "source",
                             "odi_raw": "odi"}).assign(source=lambda x: x.source.map(
                                 {"mimic_iv": "mimic_internal", "eicu": "eicu_external"}))


def age_gaps():
    p = age().pivot(index=KEYS + ["seed", "source"], columns="age_group", values="odi")
    return (p[">80"] - p["<45"]).rename("gap").reset_index().pivot(
        index=KEYS + ["seed"], columns="source", values="gap").reset_index()


@lru_cache(None)
def hospitals():
    d = _read("exp5").rename(columns={"window_h": "window", "n": "n_patients",
                                     "auc_F": "auroc_mixed", "auc_D": "auroc_disease",
                                     "internal_odi_raw": "odi_mimic", "odi_raw": "odi_eicu",
                                     "odi_shrinkage": "odi_shrinkage"})
    return d.replace([np.inf, -np.inf], np.nan)


def combo(frame):
    return frame.groupby(KEYS, sort=False).mean(numeric_only=True).reset_index()
