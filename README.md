# Medical AI Observation Dependence

本仓库分析 ICU 预测模型对观察行为、治疗和临床状态的依赖。正式结果采用按预测截点重建的成人风险集；每个截点只使用此前可获得的输入。数据来自 MIMIC-IV 2.2 和 eICU-CRD 2.0，本地原始数据不随仓库发布。

## 正式方案

- 任务：ICU Death、Hospital Death、AKI、Deterioration、Escalation。Deterioration 是操作性恶化终点，不表示脓毒症诊断。
- 算法：Logistic Regression、Random Forest、XGBoost、LightGBM、MLP；窗口 3/6/12/18/24 h；种子 42–46。
- 四输入组：`clinical_state`、`observation`、`clinical_state_treatment`、`full_model`。
- 三增益：`observation_gain = AUROC(full_model) − AUROC(clinical_state_treatment)`；`treatment_gain = AUROC(clinical_state_treatment) − AUROC(clinical_state)`；`combined_gain = AUROC(full_model) − AUROC(clinical_state)`。
- 概率误差：`brier_improvement = Brier(clinical_state_treatment) − Brier(full_model)`；正值表示总体 Brier 误差减小。
- 外部主分析只将两项死亡结局解释为跨医院验证；其余三项完整保留为探索性结果。

[证据与未确定事项](docs/audit/landmark_v2_uncertainties.md) 逐项记录原始字段能支持的结论及不能直接证明的时间语义。正式定义、bootstrap、患者划分见 `outputs/analysis_protocol.json`；逐窗口人数变化见 `outputs/data_impact.csv`。

## 运行

在 `code/` 目录、安装 `requirements.txt` 后执行。原始文件放在配置中的 `data/raw/` 路径。

```bash
python scripts/build_poe_times.py
python scripts/build_cohort.py --config configs/default.yaml --dataset mimic --output-dir data/processed/landmark_v2_staging
python scripts/build_cohort.py --config configs/default.yaml --dataset eicu --output-dir data/processed/landmark_v2_staging
python scripts/run_formal.py --stage run --workers 8
```

长任务可按 `prepare`、`plan`、`train`、`infer`、`evaluate`、`collect`、`validate`、`promote` 分段运行。`train` 按每个具体输入组的 `fit_id` 与模型、预处理器 SHA 续跑；`infer` 只读取已完成权重；`evaluate/collect/validate` 不隐式训练或推理。正式总流程在 staging 内完成校验后再替换正式产物。

## 结果

- `outputs/exp1/model_metrics.csv.gz`、`gain_metrics.csv.gz`：逐配置×输入组指标和配对增益。
- `outputs/exp2/`：固定测试集匹配、平衡、入院情境、诊断录入扩展及独立状态增强结果。24 h Hospital Death 的最新生命体征对照额外拟合 20 个模型；主模型网格为 2,500 个组拟合。
- `outputs/exp3/`：内部来源分解与跨医院转移变化。
- `outputs/exp4/`：年龄分组及年龄差转移。
- `outputs/exp5/`：逐医院表现、反转比例与住院死亡 24 h 概率误差。
- `outputs/models/`：本地模型、预处理器、完整内部和外部患者概率与来源 manifest；不推送原始数据或患者级产物。

所有结果保留完整计算精度，单类别 AUROC 记为不可评估，Brier 可独立计算。五个种子是重复拟合，不是五倍独立患者。`fig/` 和论文在本轮保持原样。
