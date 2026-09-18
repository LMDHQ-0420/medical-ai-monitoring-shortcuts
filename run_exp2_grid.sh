#!/usr/bin/env bash
# run_exp2_grid.sh — exp2 (severity control ablation) dispatcher
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
# output subdir. Override with: SEED=43 ./run_exp2_grid.sh
read -r -a SEEDS <<< "${SEEDS:-${SEED:-42}}"
PSM_ONLY="${PSM_ONLY:-0}"

MODELS=(lr rf xgb lgbm gru mlp)
WINDOWS=(3 6 12 18 24)
GPUS=(0 1 2 3)
CONCURRENCY="${CONCURRENCY:-12}"

QUEUE=()
for seed in "${SEEDS[@]}"; do
    for m in "${MODELS[@]}"; do
        for w in "${WINDOWS[@]}"; do
            QUEUE+=("$m,$w,$seed")
        done
    done
done

LOG_DIR="$PROJECT_DIR/logs/exp2_grid"
mkdir -p "$LOG_DIR"

HEAD=0
GPU_PTR=0
RUNNING=0
STATUS=0

launch() {
    local item="${QUEUE[$HEAD]}"
    HEAD=$((HEAD + 1))
    local model win seed
    IFS=, read -r model win seed <<< "$item"
    mkdir -p "$LOG_DIR/seed$seed"
    local log="$LOG_DIR/seed$seed/${model}_${win}h.log"
    local cmd="cd '$PROJECT_DIR' && python3 scripts/exp2_ablation_severity.py --config configs/default.yaml --model $model --window $win --seed $seed"
    if [[ "$PSM_ONLY" == "1" ]]; then
        cmd+=" --psm-only"
    fi

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

echo "Starting exp2 dispatcher | tasks: ${#QUEUE[@]} | concurrency: $CONCURRENCY"
echo "Logs -> $LOG_DIR"

while (( HEAD < ${#QUEUE[@]} )) && (( RUNNING < CONCURRENCY )); do
    launch
done

while (( RUNNING > 0 )); do
    if ! wait -n; then
        STATUS=1
    fi
    RUNNING=$((RUNNING - 1))
    if (( HEAD < ${#QUEUE[@]} )); then
        launch
    fi
done

echo "All exp2 tasks done (status=$STATUS). Logs: $LOG_DIR"
exit "$STATUS"
