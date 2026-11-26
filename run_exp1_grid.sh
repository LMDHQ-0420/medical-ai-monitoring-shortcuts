#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# SEED 可选；不设置时运行协议中全部五个种子。
if [[ -n "${SEED:-}" ]]; then
  exec python scripts/exp1_train_grid.py --seed "$SEED" "$@"
fi
exec python scripts/exp1_train_grid.py "$@"
