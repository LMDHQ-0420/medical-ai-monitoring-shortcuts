# Medical AI Observation Dependence

Code for the PNAS study: *"Clinical monitoring patterns can mislead medical AI training"*

GitHub repository name: `medical-ai-monitoring-shortcuts`.

## Overview

This codebase quantifies how much medical AI models depend on clinical observation behaviour (how intensively a patient is monitored) rather than pure disease state, via:

- **ODI** (Observation Dependence Index): AUROC(M-Model) − AUROC(D-Model)

Five experiments (exp1–exp5) test this metric's magnitude, robustness, and fairness implications.

---

## Environment Setup

```bash
conda create -n medical-ai-monitoring-shortcuts python=3.10 -y
conda activate medical-ai-monitoring-shortcuts

# Install PyTorch (adjust CUDA version as needed)
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
# Run from the repository root
pip install -r requirements.txt
```

---

## Data Preparation

Both datasets require PhysioNet CITI training certification.
Apply at: https://physionet.org/settings/credentialing/

Place downloaded files at the following paths (relative to the repository root):

```
data/
└── raw/
    ├── mimic_iv/
    │   ├── hosp/
    │   │   ├── patients.csv.gz
    │   │   ├── admissions.csv.gz
    │   │   ├── labevents.csv.gz
    │   │   ├── prescriptions.csv.gz
    │   │   └── diagnoses_icd.csv.gz
    │   └── icu/
    │       ├── icustays.csv.gz
    │       ├── chartevents.csv.gz     # ~30 GB
    │       ├── inputevents.csv.gz
    │       └── procedureevents.csv.gz
    └── eicu/
        ├── patient.csv.gz
        ├── lab.csv.gz
        ├── vitalPeriodic.csv.gz
        ├── medication.csv.gz
        ├── apachePatientResult.csv.gz
        └── hospital.csv.gz
```

Download commands (replace `{username}` with your PhysioNet account):

```bash
# MIMIC-IV v2.2
wget -r -N -c -np --user {username} --ask-password \
  https://physionet.org/files/mimiciv/2.2/ -P data/raw/mimic_iv/

# eICU-CRD v2.0
wget -r -N -c -np --user {username} --ask-password \
  https://physionet.org/files/eicu-crd/2.0/ -P data/raw/eicu/
```

---

## Running the Pipeline

All commands are run from the repository root (the local `code/` directory).

### Step 1: Build Cohorts

```bash
# Full run
python scripts/build_cohort.py --config configs/default.yaml

# Smoke test (500 stays, verify column names)
python scripts/build_cohort.py --smoke-test

# Single dataset
python scripts/build_cohort.py --dataset mimic
python scripts/build_cohort.py --dataset eicu
```

### 正式实验

所有命令从 `code/` 运行。五模型主分析与 GRU-seq 在实验 1 拟合 D、O、D+T、F，实验 2–5 复用相同的正式权重。

```bash
python scripts/exp1_train_grid.py
python scripts/exp2_ablation_severity.py
python scripts/exp3_ablation_grouping.py
python scripts/exp4_age_odi_transfer.py
```

实验 4 与实验 5 共用评估入口；运行最后一条命令会生成两者的汇总表。也可使用 `python scripts/exp5_external_validation.py`。正式结果直接写到 `outputs/exp1/` 至 `outputs/exp5/`，共享权重及完整测试预测保存在 `outputs/models/`。单格续跑可使用实验 1 的 `--task`、`--model`、`--window` 和 `--seed` 参数。

### 图件

```bash
python fig/build_delivery.py  # 14 个独立面板与图例，PDF/SVG/PNG
```

绘图只读取 `outputs/exp1/` 至 `outputs/exp5/` 的正式表，最终图件保存在 `fig/fig1b/`、`fig/fig1c/` 至 `fig/fig5c/` 等分面板目录中。历史 `results/` 已删除；正式绘图只使用 `outputs/`。`fig/style.py` 统一独立面板的尺寸、字体和颜色。

---

## 项目目录

`src/data/` 负责队列与真实逐小时序列，`src/models/` 保存模型实现，`src/exp1/` 至 `src/exp5/` 分别保存五个实验的方法。原有 `scripts/build_cohort.py` 与 `scripts/exp1_train_grid.py` 至 `scripts/exp5_external_validation.py` 是正式入口。五个实验复用实验 1 保存的模型；D、O、D+T、F 四组权重只保存一份。GRU-seq 单独运行和汇总。

`data/raw/` 与 `data/processed/` 保存输入及处理数据。当前正式结果直接在 `outputs/`：

```text
outputs/
├── analysis_protocol.json
├── models/
│   ├── tabular/{task}/{model}/seed{seed}/{window}h/
│   └── gru_seq/{task}/seed{seed}/{window}h/
├── exp1/    # 主模型比较
├── exp2/    # 截点前临床状态匹配与情境分层
├── exp3/    # 共享完整模型的治疗重分组
├── exp4/    # 年龄分层
└── exp5/    # 外部医院验证
```

每个正式格点保留四组权重、必要预处理、完整内部测试预测及 `manifest.json`。`outputs/` 不保存绘图代码、生成图件、ZIP、日志或中间 bootstrap 缓存。论文 `docs/manuscripts/` 保持原稿，`fig/` 已按新结果重新生成。正式结果入口为 `outputs/exp1/` 至 `outputs/exp5/`，共享权重与患者级预测位于 `outputs/models/`。
