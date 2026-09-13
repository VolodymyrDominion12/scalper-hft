#!/usr/bin/env bash
# Синхронізація read-only копій paper-журналів із VPS → results/vps/.
#
# Навіщо окремий скрипт: дашборд і `paper-audit` читають SQLite локально, а
# журнали живуть на VPS і пишуться живим WAL-процесом. Копіювати «сирі»
# .sqlite/.sqlite-wal з-під працюючого бота не можна (рвана БД), тому віддалена
# сторона робить консистентний знімок через sqlite3 backup API і вже його
# віддає сюди.
#
# На VPS нічого не змінюється: тільки читання + тимчасовий каталог
# results/.vps_sync_tmp, який видаляється в кінці (--keep-tmp лишає для діагностики).
# Жодних записів у .env, systemd-юніти чи control.json.
#
# Використання:
#   bash scripts/sync_vps_paper.sh
#   VPS_PATH=/home/tradebot/scalper-hft LOCAL_DIR=results/vps bash scripts/sync_vps_paper.sh --keep-tmp
set -euo pipefail

VPS_USER="${VPS_USER:-tradebot}"
VPS_HOST="${VPS_HOST:-46.36.216.98}"
VPS_PORT="${VPS_PORT:-22}"
VPS_PATH="${VPS_PATH:-/home/tradebot/scalper-hft}"
SSH_KEY="${SSH_KEY:-}"
LOCAL_DIR="${LOCAL_DIR:-results/vps}"
REMOTE_TMP="${REMOTE_TMP:-results/.vps_sync_tmp}"
KEEP_TMP=0
for arg in "$@"; do
    case "$arg" in
        --keep-tmp) KEEP_TMP=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Невідомий аргумент: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -p "$VPS_PORT")
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS+=(-i "$SSH_KEY")
fi
REMOTE="${VPS_USER}@${VPS_HOST}"

log() { printf '[sync-vps-paper] %s\n' "$*" >&2; }

# ── Фаза 1: на VPS робимо консистентні знімки й збираємо метадані ───────────────
log "знімок на ${REMOTE}:${VPS_PATH} …"
MANIFEST_RAW="$(
    ssh "${SSH_OPTS[@]}" "$REMOTE" \
        "cd '$VPS_PATH' && VPS_PATH='$VPS_PATH' REMOTE_TMP='$REMOTE_TMP' \
         \$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 ) -" <<'PY'
"""Консистентні знімки paper-SQLite + метадані у stdout (JSON одним рядком)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys

repo = os.environ["VPS_PATH"]
tmp_rel = os.environ["REMOTE_TMP"]
results = os.path.join(repo, "results")
tmp = os.path.join(repo, tmp_rel)

DATABASES = [
    "paper_pairs.sqlite",
    "paper_ts_momentum_1d.sqlite",
    "paper_ts_momentum_4h.sqlite",
]
PLAIN_FILES = ["control.json", "audit_verdicts.jsonl"]
UNITS = [
    "scalper-paper-pairs",
    "scalper-paper-tsmom@1d",
    "scalper-paper-tsmom@4h",
]
COUNTED_TABLES = ("equity", "trades", "positions", "orders", "accounts", "bots")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_db(src_path: str, dst_path: str) -> None:
    """Консистентний знімок живого WAL-журналу через backup API."""
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True, timeout=20.0)
    try:
        src.execute("PRAGMA busy_timeout=20000")
        dst = sqlite3.connect(dst_path)
        try:
            with dst:
                src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def db_stats(path: str) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out: dict = {"rows": {}, "last_equity_ts": None, "last_snapshot_ts": None}
    try:
        for table in COUNTED_TABLES:
            try:
                out["rows"][table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                out["rows"][table] = None
        for key, sql in (
            ("last_equity_ts", "SELECT MAX(ts) FROM equity"),
            ("last_snapshot_ts", "SELECT MAX(saved_at) FROM snapshots"),
        ):
            try:
                out[key] = (con.execute(sql).fetchone() or [None])[0]
            except sqlite3.Error:
                out[key] = None
    finally:
        con.close()
    return out


def run(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip()


def unit_state(unit: str) -> dict:
    """Стан юніта + мітка часу останнього запису в journal (epoch µs → ISO)."""
    state = {
        "active": run(["systemctl", "--user", "is-active", unit]) or "unknown",
        "since": run(["systemctl", "--user", "show", "-p", "ActiveEnterTimestamp", "--value", unit]),
        "last_log_ts": None,
        "last_log": "",
    }
    raw = run(["journalctl", "--user", "-u", unit, "-n", "1", "-o", "json", "--no-pager"])
    if raw:
        try:
            entry = json.loads(raw.splitlines()[-1])
            micros = int(entry.get("__REALTIME_TIMESTAMP", 0))
            if micros:
                state["last_log_ts"] = (
                    dt.datetime.fromtimestamp(micros / 1e6, dt.UTC).isoformat(timespec="seconds")
                )
            state["last_log"] = str(entry.get("MESSAGE", ""))[:300]
        except (ValueError, json.JSONDecodeError):
            state["last_log"] = raw[:300]
    return state


def artifact(path: str) -> dict:
    st = os.stat(path)
    return {
        "bytes": st.st_size,
        "sha256": sha256(path),
        "mtime": dt.datetime.fromtimestamp(st.st_mtime, dt.UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    manifest: dict = {
        "remote_time": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "vps": {"path": repo, "tmp": tmp_rel},
        "databases": {},
        "files": {},
        "units": {},
        "missing": [],
    }

    for name in DATABASES:
        src_path = os.path.join(results, name)
        if not os.path.exists(src_path):
            manifest["missing"].append(name)
            continue
        dst_path = os.path.join(tmp, name)
        snapshot_db(src_path, dst_path)
        entry = artifact(dst_path)
        entry.update(db_stats(dst_path))
        manifest["databases"][name] = entry

    for name in PLAIN_FILES:
        src_path = os.path.join(results, name)
        if not os.path.exists(src_path):
            continue
        dst_path = os.path.join(tmp, name)
        shutil.copy2(src_path, dst_path)
        manifest["files"][name] = artifact(dst_path)

    for unit in UNITS:
        manifest["units"][unit] = unit_state(unit)

    sys.stdout.write(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
)"

if [[ -z "$MANIFEST_RAW" ]]; then
    log "ПОМИЛКА: VPS не повернув маніфест"
    exit 1
fi

# ── Фаза 2: передача знімків ───────────────────────────────────────────────────
mkdir -p "$LOCAL_DIR"
INCOMING="$LOCAL_DIR/.incoming"
rm -rf "$INCOMING"
mkdir -p "$INCOMING"

log "передача знімків → $LOCAL_DIR …"
ssh "${SSH_OPTS[@]}" "$REMOTE" "tar -C '$VPS_PATH/$REMOTE_TMP' -czf - ." | tar -xzf - -C "$INCOMING"

if [[ "$KEEP_TMP" == "0" ]]; then
    ssh "${SSH_OPTS[@]}" "$REMOTE" "rm -rf '$VPS_PATH/$REMOTE_TMP'"
fi

# ── Фаза 3: перевірка цілісності та атомарна підміна ──────────────────────────
MANIFEST_RAW="$MANIFEST_RAW" LOCAL_DIR="$LOCAL_DIR" INCOMING="$INCOMING" \
    VPS_HOST="$VPS_HOST" VPS_PATH="$VPS_PATH" python3 - <<'PY'
"""Звірити sha256 отриманих файлів із маніфестом, підмінити атомарно."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import sys

local_dir = os.environ["LOCAL_DIR"]
incoming = os.environ["INCOMING"]
manifest = json.loads(os.environ["MANIFEST_RAW"])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(kind: str) -> list[str]:
    bad = []
    for name, meta in manifest.get(kind, {}).items():
        src = os.path.join(incoming, name)
        if not os.path.exists(src):
            bad.append(f"{name}: немає у передачі")
            continue
        got = sha256(src)
        if got != meta["sha256"]:
            bad.append(f"{name}: sha256 {got[:12]} ≠ {meta['sha256'][:12]}")
            continue
        os.replace(src, os.path.join(local_dir, name))
    return bad


problems = verify("databases") + verify("files")
if problems:
    print("[sync-vps-paper] ПОМИЛКА цілісності:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)

manifest["verified"] = True
manifest["synced_at_local"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
manifest["source"] = {"host": os.environ["VPS_HOST"], "path": os.environ["VPS_PATH"]}

target = os.path.join(local_dir, "manifest.json")
tmp_target = target + ".tmp"
with open(tmp_target, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
os.replace(tmp_target, target)

shutil.rmtree(incoming, ignore_errors=True)
print(
    "[sync-vps-paper] OK: "
    + ", ".join(f"{n} ({m['bytes']} B)" for n, m in manifest.get("databases", {}).items()),
    file=sys.stderr,
)
PY

log "готово (маніфест: $LOCAL_DIR/manifest.json)"
