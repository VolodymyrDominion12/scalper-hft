#!/usr/bin/env bash
# Ops: старт 8-тижневого Paper-Gate після тегу paper-v0.2.0 (R1+R2+R3).
set -euo pipefail

TAG="${1:-paper-v0.2.0}"
echo "Paper-Gate checklist:"
echo "  1. git tag ${TAG} && git push origin ${TAG}"
echo "  2. На VPS: ./scripts/deploy_paper.sh ${TAG}"
echo "  3. cp docs/env/vps-paper.env.example .env  (заповнити ключі)"
echo "  4. systemctl --user restart scalper-paper-pairs"
echo "  5. Щотижня: uv run python -m scalper_hft.cli paper-audit"
echo "  6. Критерій: ≥8 тижнів без tracking error vs BT"
