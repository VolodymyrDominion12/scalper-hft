#!/usr/bin/env bash
# Цикл синхронізації VPS paper-журналів для локального моніторингу.
#
# Кожні INTERVAL секунд викликає scripts/sync_vps_paper.sh і пише рядок у лог.
# Призначений для ручного запуску в tmux/терміналі або як systemd --user
# (див. docs/RUNBOOK_PAPER_MONITORING.md). VPS не змінюється: тільки читання.
#
# Використання:
#   bash scripts/vps_paper_watch.sh                 # цикл, 300 с
#   INTERVAL=60 bash scripts/vps_paper_watch.sh     # частіше (VPS отримує ssh-сесію щохвилини)
#   bash scripts/vps_paper_watch.sh --once          # один синк + статус (для cron)
set -euo pipefail

INTERVAL="${INTERVAL:-300}"
LOG_FILE="${LOG_FILE:-results/logs/vps_paper_sync.log}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$(dirname "$LOG_FILE")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON_BIN=.venv/bin/python
    else
        PYTHON_BIN=python3
    fi
fi

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE" >&2; }

run_once() {
    if ! bash scripts/sync_vps_paper.sh >>"$LOG_FILE" 2>&1; then
        log "ПОМИЛКА синку (код $?) — див. $LOG_FILE"
        return 0
    fi

    local out
    if out="$("$PYTHON_BIN" scripts/vps_paper_status.py 2>/dev/null)"; then
        log "OK — синк і всі три боти здорові"
    else
        # Статус віддає exit 1 разом із переліком проблем — виносимо його в лог.
        log "УВАГА: проблеми — $(printf '%s' "$out" | grep -A20 'Проблеми:' | tr '\n' ' ' | tr -s ' ')"
    fi
}

if [[ "${1:-}" == "--once" ]]; then
    run_once
    "$PYTHON_BIN" scripts/vps_paper_status.py
    exit $?
fi

trap 'log "зупинено"; exit 0' TERM INT
log "старт: інтервал ${INTERVAL}с, лог $LOG_FILE"
while true; do
    run_once
    sleep "$INTERVAL"
done
