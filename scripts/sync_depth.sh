#!/usr/bin/env bash
# W0-R5: rsync depth5/bookTicker з VPS на research-машину + quality-звіт.
# Не комітити parquet. Рекордер — публічний WS, без API-ключів.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat >&2 <<'EOF'
usage: scripts/sync_depth.sh [--dry-run] [user@host:path]

  rsync -avz --partial  *depth5*.parquet *bookTicker*
  validate_depth / validate_bookticker → results/quality_depth.md
  exit ≠ 0 якщо quality_ok=false

Defaults:
  source = ${VPS_USER:-tradebot}@${VPS_HOST}:${REMOTE_DIR:-~/scalper-hft/data}
  local  = ${LOCAL_DIR:-<repo>/data}
  report = ${QUALITY_OUT:-<repo>/results/quality_depth.md}
EOF
}

DRY_RUN=0
SRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "невідомий прапорець: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -n "$SRC" ]]; then
        echo "зайвий аргумент: $1" >&2
        usage
        exit 1
      fi
      SRC="$1"
      shift
      ;;
  esac
done

LOCAL_DIR="${LOCAL_DIR:-$ROOT/data}"
QUALITY_OUT="${QUALITY_OUT:-$ROOT/results/quality_depth.md}"
VPS_USER="${VPS_USER:-tradebot}"
REMOTE_DIR="${REMOTE_DIR:-~/scalper-hft/data}"

if [[ -z "$SRC" ]]; then
  if [[ -z "${VPS_HOST:-}" ]]; then
    echo "Вкажіть user@host:path або export VPS_HOST=..." >&2
    usage
    exit 1
  fi
  SRC="${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}"
fi

# Лише host без ':path' → дефолтний remote dir.
if [[ "$SRC" != *:* ]]; then
  SRC="${SRC}:${REMOTE_DIR}"
fi

REMOTE_DATA="${SRC%/}"
mkdir -p "$LOCAL_DIR"

RSYNC_OPTS=(
  -avz
  --partial
  --include='*depth5*.parquet'
  --include='*bookTicker*.parquet'
  --include='*bookticker*.parquet'
  --exclude='*'
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

echo "rsync ${REMOTE_DATA}/ → ${LOCAL_DIR}/"
rsync "${RSYNC_OPTS[@]}" "${REMOTE_DATA}/" "${LOCAL_DIR}/"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "dry-run: пропуск validate_depth / validate_bookticker"
  exit 0
fi

echo "── L2 quality ──"
mkdir -p "$(dirname "$QUALITY_OUT")"
uv run python -m scalper_hft.cli depth-audit --dir "$LOCAL_DIR" --out "$QUALITY_OUT"
