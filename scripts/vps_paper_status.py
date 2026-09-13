#!/usr/bin/env python3
"""Статус VPS paper-ботів за локальним знімком `results/vps` (тільки читання).

Приклад:
    uv run python scripts/vps_paper_status.py
    uv run python scripts/vps_paper_status.py --json
    bash scripts/sync_vps_paper.sh && uv run python scripts/vps_paper_status.py

Exit code 1 — щось не гаразд (протухлий синк, юніт не active, журнал мовчить
довше за інтервал + запас). Це робить скрипт придатним для cron/monitor-алертів.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalper_hft.live.vps_paper_snapshot import (  # noqa: E402
    DEFAULT_MAX_SYNC_AGE_HOURS,
    DEFAULT_SNAPSHOT_DIR,
    build_report,
    format_table,
    pairs_gate_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Статус VPS paper-ботів за локальним знімком")
    parser.add_argument("--dir", default=str(DEFAULT_SNAPSHOT_DIR), help="Каталог знімка (results/vps)")
    parser.add_argument(
        "--max-sync-age-hours",
        type=float,
        default=DEFAULT_MAX_SYNC_AGE_HOURS,
        help="Скільки годин може жити знімок, перш ніж це вважається зламаним синком",
    )
    parser.add_argument("--json", action="store_true", help="Вивести машинозчитуваний JSON")
    args = parser.parse_args(argv)

    report = build_report(args.dir, max_sync_age_hours=args.max_sync_age_hours)
    gate_ok, gate_msg = pairs_gate_status(args.dir)

    if args.json:
        payload = {
            "snapshot_dir": str(report.snapshot_dir),
            "synced_at": None if report.synced_at is None else report.synced_at.isoformat(timespec="seconds"),
            "sync_age_hours": report.sync_age_hours,
            "healthy": report.healthy,
            "issues": report.issues,
            "control": report.control,
            "pairs_gate": {"ok": gate_ok, "message": gate_msg},
            "bots": [
                {
                    "key": snap.spec.key,
                    "label": snap.spec.label,
                    "unit": snap.spec.unit,
                    "unit_active": snap.unit_active,
                    "equity": snap.journal.equity,
                    "realized_pnl": snap.journal.realized_pnl,
                    "trades": snap.journal.trades,
                    "open_positions": snap.journal.open_positions,
                    "orders": snap.journal.orders,
                    "orders_by_status": snap.journal.orders_by_status,
                    "last_write_ts": None
                    if snap.last_write_ts is None
                    else snap.last_write_ts.isoformat(timespec="seconds"),
                    "age_hours": snap.age_hours,
                    "healthy": snap.healthy,
                    "issues": snap.issues,
                }
                for snap in report.snapshots
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_table(report))
        mark = "PASS" if gate_ok else "НЕ PASS"
        print(f"Гейт аудиту pairs (LINKUSDT/BTCUSDT 1h): {mark} — {gate_msg}")

    return 0 if report.healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
