"""ensure_klines: 1m — джерело істини, похідні ТФ не пишуться в кеш."""

from __future__ import annotations

import pandas as pd
from scalper_hft.data.access import can_derive, ensure_klines, klines_from_store
from scalper_hft.data.research import load_research_data


def _bars_df(start: str, n: int, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    close = pd.Series(range(n), index=idx, dtype=float) + 100.0
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0},
        index=idx,
    )


class _MemStore:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], pd.DataFrame] = {}

    def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
        df = self.data.get((symbol, interval))
        return None if df is None else df.copy()

    def save_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        self.data[(symbol, interval)] = df.copy()


def test_can_derive() -> None:
    assert can_derive("5m", "1m")
    assert can_derive("1h", "1m")
    assert not can_derive("1m", "1m")
    assert not can_derive("1m", "5m")
    assert not can_derive("90s", "1m")


def test_ensure_klines_derive_does_not_persist_target(monkeypatch) -> None:
    from scalper_hft.data import access as acc
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:30")
    store = _MemStore()
    store.save_klines("BTCUSDT", "1m", _bars_df("2024-01-09 12:00", 24 * 60))
    monkeypatch.setattr(acc, "get_store", lambda: store)
    monkeypatch.setattr(dl, "get_store", lambda: store)
    monkeypatch.setattr(dl, "_utc_now", lambda: now)

    out = ensure_klines("BTCUSDT", "5m", days=1, derive=True)
    assert len(out) == (24 * 60) // 5
    assert store.load_klines("BTCUSDT", "5m") is None
    assert store.load_klines("BTCUSDT", "1m") is not None


def test_klines_from_store_resamples_1m(monkeypatch) -> None:
    from scalper_hft.data import access as acc

    store = _MemStore()
    store.save_klines("ETHUSDT", "1m", _bars_df("2024-01-01", 120))
    monkeypatch.setattr(acc, "get_store", lambda: store)
    out = klines_from_store("ETHUSDT", "5m", days=1)
    assert out is not None
    assert len(out) == 24  # 120 1m → 24 повних 5m (останній неповний відкидається? 120/5=24 exactly)
    assert store.load_klines("ETHUSDT", "5m") is None


def test_ensure_klines_readonly_uses_cache_no_network(monkeypatch) -> None:
    from scalper_hft.data import access as acc
    from scalper_hft.data import downloader as dl

    store = _MemStore()
    store.save_klines("BTCUSDT", "1m", _bars_df("2024-01-01", 120))
    monkeypatch.setattr(acc, "get_store", lambda: store)
    monkeypatch.setattr(dl, "get_store", lambda: store)

    out = ensure_klines("BTCUSDT", "5m", days=1, readonly=True)
    assert len(out) == 24  # 120 1m → 24 повних 5m
    # нічого не записано і не докачано
    assert store.load_klines("BTCUSDT", "5m") is None
    assert store.load_klines("BTCUSDT", "1m") is not None


def test_ensure_klines_readonly_raises_when_cache_empty(monkeypatch) -> None:
    import pytest
    from scalper_hft.data import access as acc
    from scalper_hft.data import downloader as dl

    store = _MemStore()
    monkeypatch.setattr(acc, "get_store", lambda: store)
    monkeypatch.setattr(dl, "get_store", lambda: store)

    with pytest.raises(RuntimeError, match="readonly"):
        ensure_klines("BTCUSDT", "1m", days=1, readonly=True)


def test_load_research_data_defaults_to_derive(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_ensure(
        symbol: str,
        interval: str,
        days: int,
        *,
        base_interval: str = "1m",
        derive: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        seen["interval"] = interval
        seen["derive"] = derive
        seen["base"] = base_interval
        return _bars_df("2024-01-01", 60)

    monkeypatch.setattr("scalper_hft.data.access.ensure_klines", fake_ensure)
    bundle = load_research_data("BTCUSDT", "15m", 7, load_l2=False, validate=False)
    assert seen["derive"] is True
    assert seen["base"] == "1m"
    assert seen["interval"] == "15m"
    assert len(bundle.klines) == 60
