#!/usr/bin/env bash
# Безперервний запис bookTicker/depth5 для накопичення OB-даних.
# Запуск: scripts/record_loop.sh BTCUSDT 120  (120 хвилин; 0 = безкінечно)
set -euo pipefail
cd "$(dirname "$0")/.."

SYMBOL=${1:-BTCUSDT}
MINUTES=${2:-120}
PY=.venv/bin/python

echo "[$(date)] Початок запису ${SYMBOL} на ${MINUTES} хв (0 = безкінечно)"
while true; do
  $PY -m scalper_hft.cli record-bookticker --symbol "$SYMBOL" --minutes "${MINUTES}" --depth 2>&1 | tail -1
  if [ "${MINUTES}" != "0" ]; then
    # --minutes 0 не підтримується клієнтом; для безкінечного циклу використовуй велике число
    break
  fi
  sleep 5
done
echo "[$(date)] Завершено"
