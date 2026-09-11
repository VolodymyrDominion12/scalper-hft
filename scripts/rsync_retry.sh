#!/usr/bin/env bash
# rsync з повтором на код 23/24: рекордер атомарно підміняє parquet,
# mmap на VPS тоді дає ENODATA (61) і rsync виходить 23 навіть після
# успішного внутрішнього retry того самого файлу.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/rsync_retry.sh [rsync-args...]

  Обгортка над rsync. Коди 23 (partial I/O) і 24 (vanished source)
  повторюються, бо depth5/bookTicker пишуться під час pull.

Env:
  RSYNC_RETRIES      всього спроб (дефолт 3)
  RSYNC_RETRY_SLEEP  пауза між спробами, сек (дефолт 2)
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -gt 0 ]] || exit 2
  exit 0
fi

MAX="${RSYNC_RETRIES:-3}"
SLEEP="${RSYNC_RETRY_SLEEP:-2}"
if ! [[ "$MAX" =~ ^[1-9][0-9]*$ ]]; then
  echo "RSYNC_RETRIES має бути додатним цілим, зараз: $MAX" >&2
  exit 2
fi

attempt=0
while true; do
  rc=0
  rsync "$@" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    exit 0
  fi
  if [[ "$rc" -ne 23 && "$rc" -ne 24 ]]; then
    exit "$rc"
  fi
  attempt=$((attempt + 1))
  if [[ "$attempt" -ge "$MAX" ]]; then
    echo "rsync: код $rc після $MAX спроб (файл на джерелі змінюється під час копіювання)." >&2
    echo "Повторіть make pull-data; локальна копія могла вже оновитись." >&2
    exit "$rc"
  fi
  echo "==> rsync код $rc (live parquet mmap/flush) — повтор ${attempt}/${MAX} через ${SLEEP}s" >&2
  sleep "$SLEEP"
done
