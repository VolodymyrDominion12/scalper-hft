"""Тести bookticker_recorder (без реального WebSocket)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scalper_hft.live.bookticker_recorder import _event_ts, _flush
from scalper_hft.live.partition_writer import partition_path


def test_event_ts_prefers_exchange_time() -> None:
    ts = _event_ts({"E": 1_700_000_000_123, "b": "1", "B": "1", "a": "2", "A": "1"})
    assert ts == pd.Timestamp("2023-11-14 22:13:20.123")


def test_event_ts_fallback_now() -> None:
    ts = _event_ts({"b": "1", "B": "1", "a": "2", "A": "1"})
    assert isinstance(ts, pd.Timestamp)


def test_flush_writes_parquet(tmp_path: Path) -> None:
    rows = [
        {"ts": pd.Timestamp("2025-01-01"), "bid": 100.0, "bid_qty": 1.0, "ask": 100.1, "ask_qty": 2.0},
        {"ts": pd.Timestamp("2025-01-01 00:00:01"), "bid": 100.0, "bid_qty": 1.5, "ask": 100.1, "ask_qty": 2.5},
    ]
    out = tmp_path / "BTCUSDT_bookTicker.parquet"
    _flush(rows, out)
    part = partition_path(out, pd.Timestamp("2025-01-01"))
    assert part.exists()
    df = pd.read_parquet(part)
    assert len(df) == 2
    assert "bid" in df.columns


def test_flush_appends_without_duplicates(tmp_path: Path) -> None:
    out = tmp_path / "ETHUSDT_bookTicker.parquet"
    row = {"ts": pd.Timestamp("2025-01-01"), "bid": 50.0, "bid_qty": 1.0, "ask": 50.1, "ask_qty": 1.0}
    _flush([row], out)
    _flush([row], out)
    part = partition_path(out, pd.Timestamp("2025-01-01"))
    df = pd.read_parquet(part)
    assert len(df) == 1


def test_record_bookticker_sync_wrapper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """record_bookticker → asyncio.run(_record_symbol)."""
    import dataclasses

    from scalper_hft.config import get_settings, set_settings
    from scalper_hft.live.bookticker_recorder import record_bookticker

    orig = get_settings()
    try:
        set_settings(dataclasses.replace(orig, data_dir=tmp_path))
        monkeypatch.setattr(
            "scalper_hft.live.bookticker_recorder.asyncio.run",
            lambda coro: 42,
        )
        n = record_bookticker("BTCUSDT", minutes=0, data_dir=tmp_path)
        assert n == 42
    finally:
        set_settings(orig)
