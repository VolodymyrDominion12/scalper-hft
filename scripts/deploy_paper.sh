#!/usr/bin/env bash
# Оновити paper-бота на VPS до конкретного git-тегу.
# Не робить git pull main і не чіпає .env.
set -euo pipefail

TAG=${1:?usage: scripts/deploy_paper.sh paper-vX.Y.Z}
ROOT=${SCALPER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "немає $ROOT/.env — створи з .env.example, скрипт його не пише" >&2
  exit 1
fi

git fetch --tags --force
git checkout "$TAG"
if command -v uv >/dev/null 2>&1; then
  uv sync --frozen
fi

systemctl --user restart scalper-paper-pairs
systemctl --user --no-pager --full status scalper-paper-pairs || true
