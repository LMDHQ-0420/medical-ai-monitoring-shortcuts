#!/usr/bin/env bash
# run_exp4_grid.sh — exp4 (age-based ODI transfer, mimic_gap) dispatcher
# 6 models x 5 windows = 30 tasks; each task runs all 5 clinical tasks internally.
# 12 tasks run concurrently. As soon as one finishes, the next queued task launches.
# CPU models (lr/rf/xgb/lgbm/mlp) launch directly. gru (the only CUDA model)
# round-robins across GPUs 0/1/2/3.

set -euo pipefail

# Resolve project dir from this script's own location, and use whatever
# python/conda env is already active in the calling shell (no conda run,
# no hardcoded absolute paths) — activate your env manually before running.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Seed for train/val/test split + model init; also selects the results/.../seedN/
# output subdir. Override with: SEED=43 ./run_exp4_grid.sh
SEED="${SEED:-42}"

MODELS=(lr rf xgb lgbm gru mlp)
WINDOWS=(3 6 12 18 24)
GPUS=(0 1 2 3)
CONCURRENCY=12

QUEUE=()
for m in "${MODELS[@]}"; do
    for w in "${WINDOWS[@]}"; do
        QUEUE+=("$m,$w")
    done
done

LOG_DIR="$PROJECT_DIR/logs/exp4_grid/seed$SEED"
mkdir -p "$LOG_DIR"

HEAD=0
GPU_PTR=0
RUNNING=0

launch() {
    local item="${QUEUE[$HEAD]}"
    HEAD=$((HEAD + 1))
    local model="${item%,*}"
    local win="${item#*,}"
    local log="$LOG_DIR/${model}_${win}h.log"
    local cmd="cd '$PROJECT_DIR' && python3 scripts/exp4_age_odi_transfer.py --config configs/default.yaml --model $model --window $win --seed $SEED"

    if [[ "$model" == "gru" ]]; then
        local gpu="${GPUS[$GPU_PTR]}"
        GPU_PTR=$(( (GPU_PTR + 1) % ${#GPUS[@]} ))
        echo "[dispatch] GPU $gpu <- model=$model window=${win}h (log: $log)"
        CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" >"$log" 2>&1 &
    else
        echo "[dispatch] CPU <- model=$model window=${win}h (log: $log)"
        bash -c "$cmd" >"$log" 2>&1 &
    fi
    RUNNING=$((RUNNING + 1))
}

echo "Starting exp4 dispatcher | tasks: ${#QUEUE[@]} | concurrency: $CONCURRENCY"
echo "Logs -> $LOG_DIR"

while (( HEAD < ${#QUEUE[@]} )) && (( RUNNING < CONCURRENCY )); do
    launch
done

while (( RUNNING > 0 )); do
    wait -n
    RUNNING=$((RUNNING - 1))
    if (( HEAD < ${#QUEUE[@]} )); then
        launch
    fi
done

echo "All exp4 tasks done. Logs: $LOG_DIR"
