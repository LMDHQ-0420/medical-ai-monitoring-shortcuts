# PNAS 修订运行说明

本目录为当前代码的正式入口，处理队列和真实逐小时序列直接覆盖保存在 `data/processed/`。历史 `results/` 保留只读。本轮每次运行的审查、统计、图件和证据集中在 `outputs/pnas_revision/<run_id>/`；本次 `run_id=20261125`。源代码没有数据版本后缀或并行复制。

## 顺序

1. `python scripts/pnas_revision/audit.py --run-dir outputs/pnas_revision/20261125` 核查原始文件、队列、划分和旧结果异常。大文件的路径、大小、修改时间指纹与处理文件完整 SHA-256 在 `input_inventory.csv` 中分开标识。
2. `python scripts/pnas_revision/repair_cutoff_treatment.py --run-dir outputs/pnas_revision/20261125` 用原始事件按截点重算治疗和处方，原子覆盖处理队列。温度统一为摄氏的修复规则在 `mimic_cohort.py`、`eicu_cohort.py` 以及 `temperature_repair_manifest.json` 中。
3. `python scripts/pnas_revision/build_mimic_sequence.py --run-dir outputs/pnas_revision/20261125` 与 `python scripts/pnas_revision/build_eicu_sequence.py --run-dir outputs/pnas_revision/20261125` 从原始时间事件构建真实逐小时序列。必须核对 `data/processed/*_manifest.json` 的单位、时间和哈希。
4. `python scripts/pnas_revision/rerun_plan.py --run-dir outputs/pnas_revision/20261125` 逐格决定续跑；`python scripts/pnas_revision/dispatch_tabular.py --run-dir outputs/pnas_revision/20261125 --models lr rf xgb lgbm mlp --workers 4` 跑五模型；`python scripts/pnas_revision/dispatch_gru_seq.py --run-dir outputs/pnas_revision/20261125 --gpus 0 1 2 3 --workers 4` 跑独立 GRU-seq。
5. `dispatch_postprocess.py` 与 `dispatch_gru_postprocess.py` 复用已拟合模型做年龄与 eICU 外部分析。正式汇总分别运行 `aggregate.py`、`aggregate_subgroups.py`、`aggregate_gru_subgroups.py`。`bootstrap_intervals.py` 对同一患者索引复算配对 AUROC 差值的百分位区间。
6. 所有完整输入校验通过后运行 `python fig/pnas_plots.py --run-dir outputs/pnas_revision/20261125`。匹配和诊断科学定义未固定时 Fig. 2 保持未渲染。`make_evidence.py` 打包少量去标识完整预测与审查文件，目标小于 20 MiB，硬上限 30 MiB。

## 固定口径与限制

D=窗口内 14 项临床测量首值及年龄、性别、入院来源；O=30 项显式观察过程特征；T=3 项治疗标记；F=D+O+T。原有全 stay SOFA、完整住院诊断和合并症字段留在 parquet 供审计，但不进入活动 D/F。历史单步 `gru` 已从活动模型入口退休，`gru_seq` 接收真实 `[batch,time,features]`。

MIMIC 与 eICU 的 AKI、Deterioration、Escalation 定义、死亡截点风险人群以及严重度匹配所需截点前变量仍需科学口径确定。外部结果目前是描述性、探索性比较；不得从 ODI 或 `reference_gain` 推断观察或治疗的因果效应。详细阻塞见 `outputs/pnas_revision/20261125/data_audit_report.md` 和 `revision_status.json`。
