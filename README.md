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
conda create -n "zw@PNAS" python=3.10 -y
conda activate "zw@PNAS"

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

Place downloaded files at the following paths (relative to `code/`):

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

### exp1: Main ODI Grid

Trains D-Model / O-Model / M-Model for every task × model × time window.

```bash
# Full run (all tasks × models × windows) — takes several hours
python scripts/exp1_train_grid.py --config configs/default.yaml

# Single combination
python scripts/exp1_train_grid.py --task icu_mortality --model xgb --window 24

# Smoke test (LR, first task, 24h window, 500 samples)
python scripts/exp1_train_grid.py --smoke-test

# GPU-aware grid dispatcher (all 5 tasks x 5 windows)
./run_exp1_grid.sh
```

### exp2: Severity Control Ablation

Tests whether ODI survives PSM on SOFA, diagnosis-category stratification, and SOFA-adjusted O-Models.

```bash
python scripts/exp2_ablation_severity.py --config configs/default.yaml --model xgb --window 24

# Recompute PSM only and overwrite existing PSM results (preserve other analyses)
python scripts/exp2_ablation_severity.py --psm-only --seed 42

# GPU-aware grid dispatcher (6 models x 5 windows)
./run_exp2_grid.sh

# Recompute only PSM for all five seeds, overwriting existing results
SEEDS="42 43 44 45 46" PSM_ONLY=1 ./run_exp2_grid.sh
```

Existing exp2 result files are overwritten when rerun. PSM diagnostics include
matching counts, standardized mean differences, propensity-score summaries, and
train/validation/test sizes.

### exp3: Variable Grouping Sensitivity Ablation

Tests whether ODI survives reassigning treatment/medication variables between D-Model and O-Model.

```bash
python scripts/exp3_ablation_grouping.py --config configs/default.yaml --model xgb --window 24

# GPU-aware grid dispatcher (6 models x 5 windows)
./run_exp3_grid.sh
```

### exp4: Cross-Hospital Transfer Failure (odi_shrinkage)

Trains on MIMIC-IV, evaluates per-hospital ODI shrinkage on eICU.

```bash
python scripts/exp4_external_validation.py --config configs/default.yaml --model xgb --window 24

# GPU-aware grid dispatcher (6 models x 5 windows)
./run_exp4_grid.sh
```

### exp5: Age-Based ODI Inequity (mimic_gap)

Computes ODI by age group on MIMIC-IV and pooled eICU; core statistic is mimic_gap = ODI(>80) − ODI(<45).

```bash
python scripts/exp5_age_odi_transfer.py --config configs/default.yaml --model xgb --window 24

# GPU-aware grid dispatcher (6 models x 5 windows)
./run_exp5_grid.sh
```

---

## Repository Contents

The repository includes experiment results (including per-patient prediction CSVs).
Datasets under `data/`, runtime logs, and Python caches are excluded by `.gitignore`.
No model weights are currently saved on disk. Future checkpoints should be placed
outside `data/` (for example, under `checkpoints/`) so they can be tracked.

## Output Structure

Each experiment's results live under `results/exp{1-5}/`, structured as
`{task}/{feature_group}/{model}/{window}h_*`. exp1 has three feature groups
(disease/observation/mixed); exp2-5 only train disease and mixed models to
compute ODI as a difference, so they only have `disease/` and `mixed/`.

```
results/
├── exp1/
│   └── {task}/{disease,observation,mixed}/{lr,rf,xgb,lgbm,gru,mlp}/{window}h_{metrics,predictions}.{json,csv}
├── exp2/
│   └── {task}/mixed/{lr,rf,xgb,lgbm,gru,mlp}/{window}h_severity_control.json
├── exp3/
│   └── {task}/mixed/{lr,rf,xgb,lgbm,gru,mlp}/{window}h_grouping_ablation.json
├── exp4/
│   ├── {task}/mixed/{lr,rf,xgb,lgbm,gru,mlp}/{window}h_by_hospital.csv
│   └── pooled/mixed/{lr,rf,xgb,lgbm,gru,mlp}/{window}h_by_hospital.csv   # pooled across tasks
└── exp5/
    ├── {task}/mixed/{lr,rf,xgb,lgbm,gru,mlp}/{window}h_odi_by_age_transfer.csv
    └── summary/{lr,rf,xgb,lgbm,gru,mlp}/{window}h_odi_age_transfer_summary.json  # mimic_gap per task
```

---

## Project Structure

```
medical-ai-monitoring-shortcuts/
├── .gitignore
├── src/
│   ├── data/          # Cohort builders, feature definitions, transforms
│   ├── models/        # LR, XGBoost, GRU, RF, LightGBM, MLP wrappers + shared fit dispatch
│   ├── odi.py         # compute_odi() (bootstrap CI), shared by exp1/exp2/exp3
│   ├── exp1/           # trainer.py — EHRTrainer for the main ODI grid
│   ├── exp2/           # oii.py, severity_control.py — PSM / diagnosis stratification / SOFA covariate
│   ├── exp3/           # grouping.py — variable grouping variants (original/variant_A/variant_B)
│   ├── exp4/           # external_validation.py — MIMIC->eICU odi_shrinkage
│   ├── exp5/           # age_transfer.py — age-stratified ODI + transfer to eICU
│   └── utils/          # metrics, bootstrap CI, logger
├── scripts/            # build_cohort.py + exp1-5_*.py entry points
├── run_exp{1-5}_grid.sh # GPU-aware grid dispatchers for each experiment
├── tests/              # PSM matching and overwrite regression tests
├── configs/
│   └── default.yaml   # All hyperparameters
├── data/              # gitignored
├── results/           # Experiment results included in the repository
├── logs/              # gitignored
└── requirements.txt
```
