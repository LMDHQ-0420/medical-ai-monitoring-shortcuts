"""核对 Fig. 2 正式结果，写出来源清单、图注与中文论文对照。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(run: Path) -> None:
    matched = pd.read_csv(run / "matched_seed_metrics.csv.gz")
    context = pd.read_csv(run / "context_seed_metrics.csv.gz")
    diagnosis = pd.read_csv(run / "diagnosis_seed_metrics.csv.gz")
    balance = pd.read_csv(run / "fig2_matching/balance_diagnostics.csv")
    refit = pd.read_csv(run / "matched_refit_sensitivity.csv.gz")
    protocol = json.loads((run / "fig2_matching/matching_protocol.json").read_text())
    if len(matched) != 625 or len(protocol) != 25 or len(diagnosis) != 3750 or len(refit) != 25:
        raise ValueError("Fig. 2 主分析、25 个匹配划分或 eICU 诊断分层未完成")
    if (matched.n_matched != matched.n_pairs * 2).any():
        raise ValueError("匹配人数与成对关系不符")
    for col in ("odi_raw", "odi_residual"):
        expected = matched.auc_F_matched - matched["auc_" + ("D" if col == "odi_raw" else "DT") + "_matched"]
        if (matched[col + "_matched"] - expected).abs().max() > 1e-12:
            raise ValueError("两项 ODI 未复用同一份 F 预测")
    task = matched.groupby("task").agg(
        raw_original=("odi_raw_original", "mean"), raw_matched=("odi_raw_matched", "mean"),
        residual_original=("odi_residual_original", "mean"), residual_matched=("odi_residual_matched", "mean"),
        matched_n_median=("n_matched", "median"), matched_positive_median=("positive_matched", "median"),
    ).round(5)
    configurations = matched.groupby(["task", "model", "window_h"])[["odi_raw_matched", "odi_residual_matched"]].mean()
    summary = {
        "run_id": run.name, "matched_seed_cells": len(matched), "test_matching_splits": len(protocol),
        "balance_pass_splits": sum(bool(x["balance_pass"]) for x in protocol),
        "matched_pairs_min": min(x["n_pairs"] for x in protocol),
        "matched_pairs_max": max(x["n_pairs"] for x in protocol),
        "matched_raw_positive_cells": int(matched.odi_raw_matched.gt(0).sum()),
        "matched_residual_positive_cells": int(matched.odi_residual_matched.gt(0).sum()),
        "matched_raw_positive_configurations": int(configurations.odi_raw_matched.gt(0).sum()),
        "matched_residual_positive_configurations": int(configurations.odi_residual_matched.gt(0).sum()),
        "matched_raw_ci_low_positive_cells": int(matched.odi_raw_ci_low.gt(0).sum()),
        "matched_residual_ci_low_positive_cells": int(matched.odi_residual_ci_low.gt(0).sum()),
        "context_rows": len(context), "eicu_diagnosis_rows": len(diagnosis),
        "matched_refit_seed_cells": len(refit),
        "eicu_diagnosis_complete_rows": int(diagnosis.status.eq("complete").sum()),
        "task_means": task.to_dict("index"),
        "status": "complete" if all(bool(x["balance_pass"]) for x in protocol) else "complete_with_balance_limitations",
    }
    (run / "fig2_statistics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    numbers_path = run / "manuscript_numbers.json"
    numbers = json.loads(numbers_path.read_text())
    numbers["fig2"] = {key: summary[key] for key in (
        "matched_seed_cells", "test_matching_splits", "balance_pass_splits",
        "matched_pairs_min", "matched_pairs_max", "matched_raw_positive_cells",
        "matched_residual_positive_cells", "matched_raw_positive_configurations",
        "matched_residual_positive_configurations", "matched_raw_ci_low_positive_cells",
        "matched_residual_ci_low_positive_cells", "context_rows", "eicu_diagnosis_rows",
        "matched_refit_seed_cells", "task_means")}
    numbers_path.write_text(json.dumps(numbers, ensure_ascii=False, indent=2) + "\n")
    subgroup_path = run / "subgroup_status.json"
    subgroup = json.loads(subgroup_path.read_text())
    subgroup["matched"] = {"status": "complete", "seed_cells": 625,
                           "balance_pass_splits": summary["balance_pass_splits"],
                           "source": "matched_seed_metrics.csv.gz"}
    subgroup["diagnosis"] = {"status": "complete_exploratory", "seed_stratum_rows": 3750,
                             "source": "diagnosis_seed_metrics.csv.gz"}
    subgroup["context"] = {"status": "complete", "rows": summary["context_rows"],
                           "source": "context_seed_metrics.csv.gz"}
    subgroup.setdefault("blockers", {}).pop("matched", None)
    subgroup.setdefault("blockers", {}).pop("diagnosis", None)
    subgroup_path.write_text(json.dumps(subgroup, ensure_ascii=False, indent=2) + "\n")
    sources = []
    for path in [
        run / "analysis_protocol.json", run / "main_seed_metrics.csv.gz",
        *[ROOT / f"data/processed/mimic_iv_{window}h.parquet" for window in (3,6,12,18,24)],
        ROOT / "data/processed/mimic_iv_fig2_clinical_state.parquet",
        ROOT / "data/processed/mimic_iv_fig2_clinical_state_manifest.json",
        ROOT / "data/raw/eicu/eicu-collaborative-research-database-2.0/diagnosis.csv.gz",
        ROOT / "data/raw/eicu/eicu-collaborative-research-database-2.0/admissionDx.csv.gz",
        ROOT / "data/processed/eicu_24h.parquet",
        ROOT / "data/processed/eicu_fig2_diagnosis.parquet",
        ROOT / "data/processed/eicu_fig2_diagnosis_manifest.json",
        run / "fig2_field_impact_scope.csv", run / "fig2_validation.json",
        run / "fig2_matching/matching_protocol.json",
        run / "fig2_matching/matched_pairs.csv.gz", run / "fig2_matching/balance_diagnostics.csv",
        run / "matched_seed_metrics.csv.gz", run / "context_seed_metrics.csv.gz",
        run / "diagnosis_seed_metrics.csv.gz", run / "matched_refit_sensitivity.csv.gz",
        run / "fig2_extension_feasibility.json", run / "figures/fig2.pdf", run / "figures/fig2.png",
        run / "figures/extended_eicu_diagnosis.pdf", run / "figures/extended_eicu_diagnosis.png",
    ]:
        sources.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    (run / "fig2_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
    map_path = run / "figure_data_map.csv"
    mapping = pd.read_csv(map_path)
    replacements = {
        "A": ("fig2_matching/balance_diagnostics.csv", "all 25 matched test splits",
              "median absolute standardized mean difference, before vs after", "covariate across window and seed"),
        "B": ("matched_seed_metrics.csv.gz", "625 frozen tabular model seed cells",
              "AUROC(F)-AUROC(D); AUROC(F)-AUROC(D+T), original vs matched test", "seed then task"),
        "C": ("context_seed_metrics.csv.gz", "MIMIC admission source with both outcome classes",
              "AUROC(F)-AUROC(D+T), original test by admission source", "seed then task and context"),
    }
    for panel, (path, filt, formula, unit) in replacements.items():
        ix = mapping.index[(mapping.figure == "Fig2") & (mapping.panel == panel)]
        mapping.loc[ix, ["input_path", "filter", "formula", "aggregation_unit", "status"]] = [
            path, filt, formula, unit, f"rendered_formal_{run.name}"]
    if not ((mapping.figure == "ExtendedData2") & (mapping.panel == "A")).any():
        mapping.loc[len(mapping)] = ["ExtendedData2", "A", "diagnosis_seed_metrics.csv.gz",
                                     "eICU diagnosis entered before cutoff; eligible strata only",
                                     "AUROC(F)-AUROC(D+T)", "seed then task and diagnosis",
                                     f"rendered_exploratory_{run.name}"]
    mapping.to_csv(map_path, index=False)
    captions_path = run / "figure_captions_en.md"
    captions = captions_path.read_text()
    first = captions.index("**Fig. 2.")
    last = captions.index("**Fig. 3.", first)
    fig2_caption = (
        "**Fig. 2. Clinical-state matching and admission-context sensitivity analyses.** "
        "(A) Median absolute standardized differences before and after test-only matching "
        "of high- and low-observation-intensity patients, using age, sex, admission source, "
        "and the latest vital signs available before each prediction cutoff. Vital values were clipped at training-set 1st/99th percentiles for matching, and missingness was retained separately; the dashed line marks 0.1. "
        "(B) Raw AUROC gain, AUROC(F) − AUROC(D), and treatment-adjusted gain, "
        "AUROC(F) − AUROC(D+T), in the original and matched test cohorts. The fitted D, D+T, "
        "and F models are unchanged; both contrasts reuse the same F predictions. "
        "Patient-pair bootstrap intervals and event counts are supplied in the accompanying table. "
        "(C) Treatment-adjusted gain by admission source in MIMIC-IV. Categories reflect admission context, "
        "not final hospital diagnoses. An eICU analysis using only diagnoses entered before each cutoff "
        "is reported separately, with patients lacking a recorded diagnosis labeled 'Not recorded'. "
        "Comparisons are descriptive and do not identify a causal effect of observation.\n\n"
    )
    captions = captions[:first] + fig2_caption + captions[last:]
    heading = "**Extended Data Fig. 2. eICU diagnoses entered before prediction cutoff.**"
    if heading not in captions:
        captions += ("\n" + heading + " The treatment-adjusted AUROC gain is shown for diagnoses "
                     "recorded before each landmark using diagnosisOffset or admitDxEnteredOffset. "
                     "Named categories may overlap; 'Not recorded' does not mean absence of disease. "
                     "Cross-database outcome definitions are not fully harmonized, so this is exploratory.\n")
    captions_path.write_text(captions)
    table_rows = ["| 任务 | 原始 ODI：原测试 | 原始 ODI：匹配 | 残余 ODI：原测试 | 残余 ODI：匹配 | 匹配人数中位数 | 阳性中位数 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in task.iterrows():
        table_rows.append(f"| {name} | {row.raw_original:.5f} | {row.raw_matched:.5f} | "
                          f"{row.residual_original:.5f} | {row.residual_matched:.5f} | "
                          f"{row.matched_n_median:.0f} | {row.matched_positive_median:.0f} |")
    table = "\n".join(table_rows)
    warning = "全部 25 个测试划分达到预设绝对 SMD ≤0.1 的平衡标准。" if summary["balance_pass_splits"] == 25 else (
        f"仅 {summary['balance_pass_splits']}/25 个测试划分达到全部协变量绝对 SMD ≤0.1；"
        "其余结果是平衡不足的敏感性描述，不能声称已经充分控制临床状态。")
    report = f"""# Fig. 2 截点前临床状态匹配与情境分层修订报告

运行标识：`{run.name}`。主模型实际输入审查：表格模型 D/D+T/F 共 1,875 个模型组、GRU 对应 375 个模型组全部合格；问题 SOFA、完整住院诊断和同源合并症均未进入拟合输入。因这些字段导致的最小重跑范围为 0。逐格证据见 `fig2_field_impact_scope.csv`。

截点前临床状态从 MIMIC 原始 `charttime`、`storetime` 缓存提取；最近可用值按 `max(charttime,storetime)` 严格早于截点确定。年龄、性别、入院来源和六项常规生命体征用于核心匹配；生命体征极端值按训练集第 1／99 百分位裁剪，原始测量和时间另存，缺失以独立指示表示，不视为正常值。O 的 min-max 标准化、高低阈值与倾向模型只由训练集拟合；匹配只在原测试集内进行，不读结局或 ODI。高低观察强度按性别、入院来源精确匹配，并限制年龄和倾向分数差。配对关系、患者键、平衡、缺失和固定模型预测分别见 `fig2_matching/`。{warning}

25 个窗口×种子测试划分的匹配对数范围为 {summary['matched_pairs_min']}–{summary['matched_pairs_max']}。625 个模型格点全部在同一窗口和种子的匹配患者上重评估；匹配对子自助抽样区间分别记录于主匹配表。两项 ODI 共用保存的 F 预测。表格 D 模型仍用原协议中的窗口内首测值，匹配协变量则为截点前最后可用值；匹配不会改写模型输入。原测试集与匹配测试集的任务均值如下（AUROC 差值，先按种子格点平均）：

{table}

独立的匹配训练重训敏感性分析仅覆盖 24h、LR、五任务×五种子，共 {len(refit)} 个格点；另列于 `matched_refit_sensitivity.csv.gz`，不进入 Fig. 2B 的冻结模型均值。训练集匹配平衡另见 `fig2_refit_sensitivity/train_balance_seed*.csv`。

匹配后原始 ODI 为正 {summary['matched_raw_positive_cells']}/625，残余 ODI 为正 {summary['matched_residual_positive_cells']}/625；对应匹配对 bootstrap 95% 百分位区间下界大于 0 的格点分别为 {summary['matched_raw_ci_low_positive_cells']}/625、{summary['matched_residual_ci_low_positive_cells']}/625。这些格点重复使用患者，不是独立样本或多重比较校正后的比例。每格点人数、事件、未匹配数和预测文件路径见 `matched_seed_metrics.csv.gz`；`fig2_validation.json` 记录 625 份完整匹配预测的 SHA-256、配对键与标签校验。

Fig. 2C 改为 MIMIC 入院来源情境分层，避免将最终住院诊断误作早期已知；原始和残余 ODI 均在 `context_seed_metrics.csv.gz`。eICU 另按 `diagnosisOffset` 与 `admitDxEnteredOffset` 筛选截点前诊断，共 {summary['eicu_diagnosis_complete_rows']}/{summary['eicu_diagnosis_rows']} 个有效分层格点；`not_recorded` 表示截点前没有已录入诊断，不是没有疾病。命名类别可重叠。跨库结局定义未统一，这些外部结果仍为探索性。

GCS 与化验扩展匹配未混入核心匹配：GCS 的截点前逐事件与覆盖率尚未形成完整的同口径提取表；化验结果虽在 GRU 数据中按 `max(charttime,storetime)` 归入小时格，但精确采样与可用时间未随该汇总保留。需另行生成完整时间审计后再开展扩展匹配。
"""
    task_ci = ["| 任务 | 残余 ODI 为正格点 | 配对区间下界 >0 格点 |",
               "|---|---:|---:|"]
    for name, group in matched.groupby("task"):
        task_ci.append(f"| {name} | {int(group.odi_residual_matched.gt(0).sum())}/125 | "
                       f"{int(group.odi_residual_ci_low.gt(0).sum())}/125 |")
    infection = diagnosis.loc[(diagnosis.task == "sepsis_deterioration") &
                              (diagnosis.diagnosis == "infection"), "odi_residual"].mean()
    unrecorded = diagnosis.loc[(diagnosis.task == "escalation") &
                               (diagnosis.diagnosis == "not_recorded"), "odi_residual"].mean()
    report += ("\n## 结论与独立敏感性分析\n\n"
               f"匹配后 {summary['matched_raw_positive_configurations']}/125 个五种子配置的原始 ODI 均值为正，"
               f"{summary['matched_residual_positive_configurations']}/125 个配置的残余 ODI 均值为正；"
               "残余 ODI 不能概括为每格显著。匹配前后比较的是同一批冻结模型在原测试集与匹配测试子集"
               "的表现；两者不是独立患者。匹配对子 bootstrap 仅为子集内 D、D+T、F 的配对 AUROC 差值给出区间。\n\n"
               + "\n".join(task_ci) + "\n\n"
               f"独立匹配训练重训仅覆盖 LR、24h、五任务×五种子，共 {len(refit)} 个格点；"
               f"残余 ODI {int(refit.odi_residual.gt(0).sum())}/{len(refit)} 为正，"
               f"区间下界 {int(refit.odi_residual_ci_low.gt(0).sum())}/{len(refit)} 大于 0。"
               "它不进入 Fig.2B 冻结模型均值，也不能代表其他四模型或短窗口。\n\n"
               f"eICU 截点前诊断扩展图为探索性：Deterioration 的 infection 分层平均残余 ODI 为 {infection:.4f}，"
               f"Escalation 的 not_recorded 分层为 {unrecorded:.4f}。命名诊断类别可重叠，两库结局定义仍有差异。\n")
    (run / "fig2_revision_report_zh.md").write_text(report)
    before_after = f"""# Fig. 2 原稿结论与拟修改表述

| 原稿说法 | 本轮拟修改 | 依据 |
|---|---|---|
| 控制 SOFA 严重度后 ODI 仍显著为正 | 改为“在按预测截点前临床状态匹配的测试患者中，冻结模型的原始／残余 AUROC 增益分别为实际观测值”；逐任务报告方向和配对区间，不概括为因果控制。 | `matched_seed_metrics.csv.gz`；{warning} |
| 用最终住院诊断做早期诊断分层 | 主图改为入院来源情境分层；eICU 的截点前已录入诊断另列探索性分析。 | `context_seed_metrics.csv.gz`、`diagnosis_seed_metrics.csv.gz`。 |
| Fig. 2A 复用旧 ODI 结果，Fig. 2B/C 复用旧匹配与诊断 | A 展示临床状态平衡，B 展示原测试与匹配测试中冻结模型的原始和治疗调整后增益，C 展示入院情境。 | `figures/fig2.pdf`、`figure_data_map.csv`。 |

建议英文 Results 句式："After matching patients on clinical information available before each prediction landmark, we re-evaluated the unchanged models in the matched test subset. The raw and treatment-adjusted AUROC contrasts were reported separately, together with covariate balance and pair-cluster bootstrap intervals."

中文说明：匹配后的差值是固定模型在另一测试子集上的区分度变化；不能据此宣称观察行为或治疗造成了结局变化。
"""
    (run / "fig2_manuscript_before_after.md").write_text(before_after)
    status_path = run / "revision_status.json"
    status = json.loads(status_path.read_text())
    status["stages"]["fig2_revision"] = {
        "status": summary["status"],
        "commands": ["python scripts/pnas_revision/fig2_input_audit.py --run-dir outputs/pnas_revision/20261125",
                     "python scripts/pnas_revision/build_fig2_state.py",
                     "python scripts/pnas_revision/fig2_eicu_diagnosis.py",
                     "python scripts/pnas_revision/fig2_match.py --run-dir outputs/pnas_revision/20261125",
                     "python scripts/pnas_revision/fig2_evaluate.py --run-dir outputs/pnas_revision/20261125",
                     "python scripts/pnas_revision/fig2_eicu_strata.py --run-dir outputs/pnas_revision/20261125",
                     "python fig/pnas_plots.py --run-dir outputs/pnas_revision/20261125"],
        "outputs": ["figures/fig2.pdf", "figures/fig2.png", "matched_seed_metrics.csv.gz",
                    "context_seed_metrics.csv.gz", "diagnosis_seed_metrics.csv.gz",
                    "fig2_revision_report_zh.md", "fig2_sources.json"],
        "failure_reason": None if summary["status"] == "complete" else "部分窗口×种子临床状态平衡未达全部 |SMD|≤0.1",
        "next_command": "none for core Fig.2; GCS/lab extension requires separate exact-time extraction",
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
