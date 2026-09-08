#!/usr/bin/env bash
# Деплой безперервного запису depth5 на цільовій машині (VPS або локально).
#
# Що робить:
#   1) бере шаблон deploy/scalper-record@.service з ПОТОЧНОГО каталогу репо;
#   2) підставляє вашого користувача і поточний шлях репо (прибирає шляхи
#      машини розробника: User=volodymyr, /home/volodymyr/PycharmProjects/scalper-hft);
#   3) копіює у /etc/systemd/system/scalper-record@.service;
#   4) daemon-reload і enable --now для кожного символа зі списку.
#
# Запуск (з каталогу репо на цільовій машині; потрібен sudo):
#   sudo bash scripts/deploy_record.sh BTCUSDT LINKUSDT XRPUSDT ETHUSDT
# Без аргументів — ВЕСЬ універсум проєкту (CANONICAL_SYMBOLS, scalper_hft/symbols.py).
#
# ⚠ ДИСК: ~50–200 МБ/день НА СИМВОЛ (BTCUSDT на тесті ≈ 8–9 снапшотів/с).
#   15 символів ≈ 0.75–3 ГБ/день (~5–20 ГБ/тиждень). Якщо потрібні лише
#   ноги пар — передайте підмножину: BTCUSDT ETHUSDT XRPUSDT LINKUSDT SOLUSDT.
#
# ПЕРЕДУМОВИ на цільовій машині:
#   - репо оновлене (git pull), у каталозі репо;
#   - venv із websockets:  uv sync --extra live   (або .venv/bin/pip install websockets);
#   - scripts/record_loop.sh виконуваний:  chmod +x scripts/record_loop.sh;
#   - ключі/.env НЕ потрібні (публічний стрім Binance).
#
# Контроль після запуску:
#   systemctl status 'scalper-record@*.service' | head -30
#   journalctl -u 'scalper-record@BTCUSDT.service' -f
#   du -sh data/*depth5*.parquet

set -euo pipefail
cd "$(dirname "$0")/.."          # каталог репо
REPO="$PWD"
USER_NAME="${SUDO_USER:-$(whoami)}"
UNIT_SRC="deploy/scalper-record@.service"
UNIT_DST="/etc/systemd/system/scalper-record@.service"

if [ ! -f "$UNIT_SRC" ]; then
  echo "Помилка: немає $UNIT_SRC — оновіть репо (git pull)." >&2
  exit 1
fi

SYMBOLS=("$@")
if [ "${#SYMBOLS[@]}" -eq 0 ]; then
  # Весь універсум проєкту: scalper_hft/symbols.py CANONICAL_SYMBOLS
  # (15 ліквідних USDT-M перпів; INVUSDT/TESTUSDT — тестові, не включаємо).
  SYMBOLS=(BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT LINKUSDT ADAUSDT DOGEUSDT \
           AVAXUSDT NEARUSDT DOTUSDT ATOMUSDT UNIUSDT LTCUSDT AAVEUSDT)
fi

echo "==> Користувач: $USER_NAME | Каталог репо: $REPO"
echo "==> Генерую $UNIT_DST з підставленими шляхами"
sed -e "s|^User=.*|User=${USER_NAME}|" \
    -e "s|/home/volodymyr/PycharmProjects/scalper-hft|${REPO}|g" \
    "$UNIT_SRC" | sudo tee "$UNIT_DST" > /dev/null

echo "==> daemon-reload"
sudo systemctl daemon-reload

for sym in "${SYMBOLS[@]}"; do
  echo "==> enable --now scalper-record@${sym}.service"
  sudo systemctl enable --now "scalper-record@${sym}.service"
done

echo ""
echo "Готово. Перевірка:"
echo "  systemctl status 'scalper-record@*.service' | head -40"
echo "  sleep 30 && du -sh data/*depth5*.parquet"
