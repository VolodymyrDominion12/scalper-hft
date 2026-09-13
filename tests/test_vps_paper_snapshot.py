"""Тести знімка VPS paper-ботів (локальний моніторинг, тільки читання)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from scalper_hft.live import vps_paper_snapshot as vps
from scalper_hft.live.store import PaperStore

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _pairs_db(path: Path, ts: str, *, equity: float = 10_000.0, snapshot_ts: str | None = None) -> None:
    with PaperStore(path) as store:
        store.log_equity(pd.Timestamp(ts), "LINKUSDT/BTCUSDT", equity, cash=equity, realized_pnl=0.0)
        if snapshot_ts is not None:
            store.save_runtime({"version": 1, "equity": equity})
            # Override saved_at with the requested timestamp so tests are deterministic
            store._write("UPDATE snapshots SET saved_at = ? WHERE id = 'runtime'", (snapshot_ts,))


def _tsmom_db(path: Path, ts: str, *, symbols: int = 3, per_symbol: float = 666.0) -> None:
    total = symbols * per_symbol
    with PaperStore(path) as store:
        store.log_equity(pd.Timestamp(ts), "TSMOM_PORTFOLIO", total, cash=total, realized_pnl=0.0)
        for i in range(symbols):
            store.log_equity(pd.Timestamp(ts), f"SYM{i}USDT", per_symbol, cash=per_symbol, realized_pnl=0.0)


def _manifest(path: Path, *, synced_at: str, units_active: str = "active") -> None:
    payload = {
        "remote_time": synced_at,
        "synced_at_local": synced_at,
        "verified": True,
        "databases": {spec.db: {"bytes": 1, "sha256": "x" * 64} for spec in vps.BOTS},
        "units": {spec.unit: {"active": units_active, "since": "Sun 2026-09-13 06:28:01 UTC"} for spec in vps.BOTS},
    }
    (path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _fresh_dir(tmp_path: Path, *, pairs_ts: str = "2026-09-13 11:00:00", synced_at: str | None = None) -> Path:
    snap_dir = tmp_path / "vps"
    snap_dir.mkdir()
    _pairs_db(snap_dir / "paper_pairs.sqlite", pairs_ts, snapshot_ts=pairs_ts)
    _tsmom_db(snap_dir / "paper_ts_momentum_1d.sqlite", "2026-09-13 06:29:08.321075")
    _tsmom_db(snap_dir / "paper_ts_momentum_4h.sqlite", "2026-09-13 06:29:11.374103")
    _manifest(snap_dir, synced_at=synced_at or NOW.isoformat(timespec="seconds"))
    (snap_dir / "control.json").write_text(
        '{"pause": false, "no_new_entries": true, "flatten": false}', encoding="utf-8"
    )
    return snap_dir


def test_fresh_snapshot_is_healthy(tmp_path: Path) -> None:
    report = vps.build_report(_fresh_dir(tmp_path), now=NOW)

    assert report.healthy
    assert report.issues == []

    pairs = report.by_key("pairs_1h")
    assert pairs is not None
    assert pairs.unit_active == "active"
    assert pairs.journal.equity == 10_000.0
    assert pairs.age_hours is not None and pairs.age_hours < 1.1

    tsmom = report.by_key("tsmom_1d")
    assert tsmom is not None
    # Портфельний рядок TSMOM_PORTFOLIO, а не сума символів (3 × 666 = 1998)
    assert tsmom.journal.equity == 1998.0
    assert tsmom.journal.symbols == 4
    assert report.control == {"pause": False, "no_new_entries": True, "flatten": False}


def test_stale_journal_flags_bot(tmp_path: Path) -> None:
    snap_dir = _fresh_dir(tmp_path, pairs_ts="2026-09-13 06:00:00")
    report = vps.build_report(snap_dir, now=NOW)

    pairs = report.by_key("pairs_1h")
    assert pairs is not None
    assert not pairs.healthy
    assert any("немає записів" in msg for msg in pairs.issues)
    assert not report.healthy
    # tsmom 1d пише раз на добу — 5.5 год тиші для нього норма
    tsmom = report.by_key("tsmom_1d")
    assert tsmom is not None and tsmom.healthy


def test_inactive_unit_flags_bot(tmp_path: Path) -> None:
    snap_dir = _fresh_dir(tmp_path)
    _manifest(snap_dir, synced_at=NOW.isoformat(timespec="seconds"), units_active="failed")
    report = vps.build_report(snap_dir, now=NOW)

    pairs = report.by_key("pairs_1h")
    assert pairs is not None
    assert pairs.issues == ["юніт на VPS: failed"]


def test_stale_sync_flags_report(tmp_path: Path) -> None:
    old = datetime(2026, 9, 13, 6, 0, tzinfo=UTC).isoformat(timespec="seconds")
    report = vps.build_report(_fresh_dir(tmp_path, synced_at=old), now=NOW)

    assert any("застарілий" in msg for msg in report.issues)
    assert not report.healthy


def test_missing_snapshot_dir_reports_issues(tmp_path: Path) -> None:
    report = vps.build_report(tmp_path / "nope", now=NOW)

    assert not report.healthy
    assert any("немає каталогу знімка" in msg for msg in report.issues)
    assert all("немає копії журналу" in snap.issues for snap in report.snapshots)


def test_read_journal_does_not_create_tables(tmp_path: Path) -> None:
    """Модуль читає копію, а не мігрує її (інакше знімок «пливе»)."""
    db = tmp_path / "lean.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE equity (id INTEGER PRIMARY KEY, ts TEXT, pair TEXT, equity REAL, realized_pnl REAL)")
    con.execute("INSERT INTO equity (ts, pair, equity, realized_pnl) VALUES ('2026-09-13 11:00:00', 'X', 1.0, 0.0)")
    con.commit()
    con.close()

    stats = vps.read_journal(db, vps.BOTS_BY_KEY["pairs_1h"])
    assert stats.equity == 1.0
    assert stats.trades == 0
    assert stats.orders_by_status == {}

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert tables == {"equity"}


def test_tsmom_equity_falls_back_to_symbol_sum(tmp_path: Path) -> None:
    db = tmp_path / "no_portfolio.sqlite"
    with PaperStore(db) as store:
        for i in range(3):
            store.log_equity(pd.Timestamp("2026-09-13 06:00:00"), f"SYM{i}USDT", 100.0, cash=100.0, realized_pnl=0.0)

    stats = vps.read_journal(db, vps.BOTS_BY_KEY["tsmom_1d"])
    assert stats.equity == 300.0


def test_format_table_mentions_problems(tmp_path: Path) -> None:
    report = vps.build_report(_fresh_dir(tmp_path, pairs_ts="2026-09-13 06:00:00"), now=NOW)
    text = vps.format_table(report)

    assert "pairs_arb LINK/BTC 1h maker" in text
    assert "control.json на VPS" in text
    assert "Проблеми:" in text


def test_vps_paper_page_renders(tmp_path: Path, monkeypatch) -> None:
    """Сторінка «VPS Paper» малюється без винятків на реальному знімку-фікстурі."""
    from streamlit.testing.v1 import AppTest

    snap_dir = _fresh_dir(tmp_path)
    monkeypatch.setattr(vps, "DEFAULT_SNAPSHOT_DIR", snap_dir)

    script_path = Path(__file__).resolve().parent.parent / "scalper_hft" / "app_pages" / "vps_paper.py"
    at = AppTest.from_file(str(script_path), default_timeout=60).run()

    assert not at.exception
    assert [t.value for t in at.title] == ["VPS Paper"]
    assert "pairs_arb LINK/BTC 1h maker" in [m.label for m in at.metric]
