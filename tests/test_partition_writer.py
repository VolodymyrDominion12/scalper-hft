"""Tests for daily partition append writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scalper_hft.live.partition_writer import append_rows, partition_path


def test_partition_path_naming(tmp_path: Path) -> None:
    base = tmp_path / "BTCUSDT_bookTicker.parquet"
    day = pd.Timestamp("2024-03-15 12:34:56")
    out = partition_path(base, day)
    assert out.name == "BTCUSDT_bookTicker_20240315.parquet"
    assert out.parent == tmp_path


def test_append_rows_atomic_same_day(tmp_path: Path) -> None:
    base = tmp_path / "SYM_events.parquet"
    ts1 = pd.Timestamp("2024-01-01 10:00:00")
    ts2 = pd.Timestamp("2024-01-01 10:00:01")
    ts3 = pd.Timestamp("2024-01-01 11:00:00")

    append_rows(base, [{"ts": ts1, "bid": 1.0, "ask": 1.1}])
    part = partition_path(base, ts1)
    assert part.exists()
    assert not base.exists()

    df1 = pd.read_parquet(part)
    assert len(df1) == 1

    append_rows(base, [{"ts": ts2, "bid": 2.0, "ask": 2.1}, {"ts": ts3, "bid": 3.0, "ask": 3.1}])
    df2 = pd.read_parquet(part)
    assert len(df2) == 3
    assert float(df2["bid"].iloc[-1]) == pytest.approx(3.0)


def test_append_rows_separate_days(tmp_path: Path) -> None:
    base = tmp_path / "SYM_macro.parquet"
    d1 = pd.Timestamp("2024-02-01 23:59:00")
    d2 = pd.Timestamp("2024-02-02 00:01:00")
    append_rows(base, [{"ts": d1, "v": 1.0}])
    append_rows(base, [{"ts": d2, "v": 2.0}])
    p1 = partition_path(base, d1)
    p2 = partition_path(base, d2)
    assert p1.exists() and p2.exists()
    assert len(pd.read_parquet(p1)) == 1
    assert len(pd.read_parquet(p2)) == 1


def test_append_rows_dedupes_index(tmp_path: Path) -> None:
    base = tmp_path / "SYM_ticks.parquet"
    ts = pd.Timestamp("2024-05-05 15:00:00")
    append_rows(base, [{"ts": ts, "px": 10.0}])
    append_rows(base, [{"ts": ts, "px": 11.0}])
    part = partition_path(base, ts)
    df = pd.read_parquet(part)
    assert len(df) == 1
    assert float(df["px"].iloc[0]) == pytest.approx(11.0)
