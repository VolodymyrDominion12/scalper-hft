#!/usr/bin/env bash
# Швидкий старт: дані → бектест → аудит → звіт
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SYMBOL=${1:-BTCUSDT}
INTERVAL=${2:-5m}
DAYS=${3:-30}
STRATEGY=${4:-mean_reversion}

echo "═══ 1/4 Дані ═══"
$PY -m scalper_hft.cli download --symbol "$SYMBOL" --interval "$INTERVAL" --days "$DAYS"

echo "═══ 2/4 Бектест ═══"
$PY -m scalper_hft.cli backtest --strategy "$STRATEGY" --symbol "$SYMBOL" --interval "$INTERVAL" --days "$DAYS"

echo "═══ 3/4 Аудит перенавчання ═══"
$PY -m scalper_hft.cli overfit --strategy "$STRATEGY" --symbol "$SYMBOL" --interval "$INTERVAL" --days "$DAYS"

echo "═══ 4/4 Звіт ═══"
$PY -m scalper_hft.cli report --strategy "$STRATEGY" --symbol "$SYMBOL" --interval "$INTERVAL" --days "$DAYS"

echo "Готово. Звіт: docs/reports/${STRATEGY}_${SYMBOL}_${INTERVAL}.md"
