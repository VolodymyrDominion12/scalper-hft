"""Тести ресемплінгу OHLCV: 1m → 5m/15m/30m/1h/4h/1d та валідація меж."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.data.resample import (
    check_target_valid,
    infer_interval_minutes,
    interval_minutes,
    pandas_resample_rule,
    resample_klines,
    resample_series,
)


def _make_1m(hours: int = 72) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=hours * 60, freq="1min")
    close = np.linspace(100, 110, len(idx))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(idx), 1.0),
        },
        index=idx,
    )


@pytest.mark.parametrize(
    ("interval", "rule"),
    [("1m", "1min"), ("5m", "5min"), ("1h", "1h"), ("4h", "4h"), ("1d", "1D"), ("1w", "7D")],
)
def test_pandas_resample_rule_maps_binance_intervals(interval: str, rule: str) -> None:
    assert pandas_resample_rule(interval) == rule


@pytest.mark.parametrize(
    ("target", "expected_bars"),
    [("5m", 864), ("15m", 288), ("30m", 144), ("1h", 72), ("4h", 18), ("1d", 3)],
)
def test_resample_bar_counts(target: str, expected_bars: int) -> None:
    out = resample_klines(_make_1m(72), target)
    assert len(out) == expected_bars
    # індекс — час відкриття, кратний цільовому інтервалу
    assert out.index[0] == pd.Timestamp("2024-01-01 00:00:00")


def test_resample_ohlcv_aggregation() -> None:
    df = _make_1m()
    out = resample_klines(df, "5m")
    first = out.iloc[0]
    assert first["open"] == df["open"].iloc[0]
    assert first["close"] == df["close"].iloc[4]
    assert first["high"] == df["high"].iloc[:5].max()
    assert first["low"] == df["low"].iloc[:5].min()
    assert first["volume"] == pytest.approx(5.0)


def test_resample_drops_incomplete_last_bar() -> None:
    df = _make_1m().iloc[:-2]  # дані обриваються 23:58 → бін 23:55 неповний
    out = resample_klines(df, "5m")
    assert out.index[-1] == pd.Timestamp("2024-01-03 23:50:00")


def test_resample_keeps_complete_last_bar() -> None:
    df = _make_1m()  # останній бін 23:55 повний (23:55..23:59)
    out = resample_klines(df, "5m")
    assert out.index[-1] == pd.Timestamp("2024-01-03 23:55:00")


def test_resample_normalizes_tz() -> None:
    df = _make_1m()
    df.index = df.index.tz_localize("UTC")
    out = resample_klines(df, "1h")
    assert out.index.tz is None


def test_resample_upsampling_raises() -> None:
    df = _make_1m()
    with pytest.raises(ValueError, match="менший за джерело"):
        resample_klines(df, "30s")


def test_resample_non_multiple_raises() -> None:
    df = _make_1m()
    # 90s = 1.5 хв — не кратне 1m джерелу
    with pytest.raises(ValueError, match="не кратний"):
        resample_klines(df, "90s")


def test_resample_empty_returns_empty() -> None:
    out = resample_klines(pd.DataFrame(), "5m")
    assert out.empty


def test_check_target_valid_boundary() -> None:
    check_target_valid(1.0, 60.0)  # 1m → 1h ок
    with pytest.raises(ValueError):
        check_target_valid(5.0, 2.0)


def test_interval_minutes() -> None:
    assert interval_minutes("1m") == 1.0
    assert interval_minutes("1h") == 60.0
    assert interval_minutes("4h") == 240.0
    assert interval_minutes("1d") == 1440.0
    assert interval_minutes("15s") == pytest.approx(0.25)


def test_infer_interval_minutes() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="5min")
    assert infer_interval_minutes(idx) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        infer_interval_minutes(pd.DatetimeIndex(["2024-01-01"]))


def test_resample_series_funding() -> None:
    s = pd.Series([0.0001, 0.0002, 0.0003, 0.0004], index=pd.date_range("2024-01-01", periods=4, freq="1h"))
    out = resample_series(s, "4h", source="1h")
    assert len(out) == 1 and out.iloc[0] == pytest.approx(0.0004)  # 'last'
