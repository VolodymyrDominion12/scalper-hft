#!/usr/bin/env bash
# Paper-моніторинг ts_momentum 1d long-only (pre-registration iter10).
# Запуск: ./scripts/paper_ts_momentum.sh [--daemon]
set -euo pipefail
cd "$(dirname "$0")/.."

export DRY_RUN=true
export MAKER_EXECUTION=true
export DATA_EXCHANGE=binanceusdm

echo "=== data-audit ==="
uv run python -m scalper_hft.cli data-audit --days 1095

DAEMON_FLAG=""
if [[ "${1:-}" == "--daemon" ]]; then
  DAEMON_FLAG="--daemon"
fi

echo "=== paper-run-ts-momentum ==="
uv run python -m scalper_hft.cli paper-run-ts-momentum \
  --interval 1d \
  --sleep 3600 \
  --control results/control.json \
  --db results/paper_ts_momentum.sqlite \
  ${DAEMON_FLAG:+$DAEMON_FLAG}
