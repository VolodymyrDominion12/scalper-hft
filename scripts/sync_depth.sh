#!/usr/bin/env bash
# R5: rsync depth5/bookTicker з VPS на research-машину + quality-звіт.
set -euo pipefail

VPS_HOST="${VPS_HOST:-}"
VPS_USER="${VPS_USER:-tradebot}"
REMOTE_DIR="${REMOTE_DIR:-~/scalper-hft/data}"
LOCAL_DIR="${LOCAL_DIR:-./data}"

if [[ -z "${VPS_HOST}" ]]; then
  echo "Вкажіть VPS_HOST (напр. export VPS_HOST=46.36.216.98)" >&2
  exit 1
fi

mkdir -p "${LOCAL_DIR}"
rsync -avz --progress \
  "${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}/*depth5*.parquet" \
  "${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}/*bookTicker*.parquet" \
  "${LOCAL_DIR}/"

echo "── Quality report ──"
uv run python -m scalper_hft.cli data-audit --days 30 || true
for f in "${LOCAL_DIR}"/*_depth5.parquet; do
  [[ -f "$f" ]] || continue
  sym=$(basename "$f" _depth5.parquet)
  echo "depth5 ${sym}: $(uv run python -c "import pandas as pd; df=pd.read_parquet('${f}'); print(len(df))") rows"
done
