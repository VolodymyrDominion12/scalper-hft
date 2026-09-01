"""Інкрементальний кеш klines: префікс, дірки, хвіст; без перекачування середини."""

from __future__ import annotations

import pandas as pd
from scalper_hft.data.downloader import (
    Downloader,
    _interval_ms,
    _to_ms,
    merge_windows,
    missing_klines_windows,
)


def _bars_df(start: str, n: int, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    close = pd.Series(range(n), index=idx, dtype=float) + 100.0
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0},
        index=idx,
    )


def _rows_from_df(df: pd.DataFrame) -> list[list[float | int]]:
    return [
        [_to_ms(ts), float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume)]
        for ts, row in df.iterrows()
    ]


class _MemStore:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], pd.DataFrame] = {}

    def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
        df = self.data.get((symbol, interval))
        return None if df is None else df.copy()

    def save_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        self.data[(symbol, interval)] = df.copy()


class _FakeKlines:
    def __init__(self, rows: list[list[float | int]]) -> None:
        self.rows = sorted(rows, key=lambda r: r[0])
        self.calls: list[int] = []

    def fetch_klines(self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> list[list[float | int]]:
        self.calls.append(int(since_ms))
        return [r for r in self.rows if r[0] >= since_ms][:limit]


def test_merge_windows_joins_overlap_and_adjacent() -> None:
    assert merge_windows([(10, 20), (15, 30), (40, 50)]) == [(10, 30), (40, 50)]
    assert merge_windows([(0, 10), (10, 20)]) == [(0, 20)]
    assert merge_windows([(5, 5), (1, 0)]) == []


def test_missing_windows_empty_is_full_range() -> None:
    windows = missing_klines_windows(None, 1000, 5000, 1000)
    assert windows == [(1000, 6000)]


def test_missing_windows_prefix_gap_and_tail() -> None:
    interval_ms = 60_000
    # 00:10..00:14, дірка 00:15..00:16, 00:17..00:19
    left = _bars_df("2024-01-01 00:10", 5)
    right = _bars_df("2024-01-01 00:17", 3)
    existing = pd.concat([left, right])
    requested = _to_ms(pd.Timestamp("2024-01-01 00:00"))
    now = _to_ms(pd.Timestamp("2024-01-01 01:00"))  # хвіст застарілий
    windows = missing_klines_windows(existing, requested, now, interval_ms)
    first = _to_ms(existing.index[0])
    last = _to_ms(existing.index[-1])
    assert windows[0] == (requested, first)
    assert any(
        s == _to_ms(pd.Timestamp("2024-01-01 00:15")) and e == _to_ms(pd.Timestamp("2024-01-01 00:17"))
        for s, e in windows
    )
    assert windows[-1][0] == last
    assert windows[-1][1] > now


def test_missing_windows_no_work_when_fresh_and_complete() -> None:
    interval_ms = 60_000
    now = pd.Timestamp("2024-01-10 12:00:00")
    existing = _bars_df("2024-01-09 12:00", 24 * 60)  # до 12:00 наступного дня виключно → last=11:59
    windows = missing_klines_windows(existing, _to_ms(pd.Timestamp("2024-01-09 12:00")), _to_ms(now), interval_ms)
    assert windows == []


def test_klines_does_not_refetch_closed_middle(monkeypatch) -> None:
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)

    existing = _bars_df("2024-01-09 18:00", 18 * 60)  # 18:00 → 11:59 next day
    prefix = _bars_df("2024-01-09 12:00", 6 * 60)  # 12:00 → 17:59
    store = _MemStore()
    store.save_klines("BTCUSDT", "1m", existing)
    client = _FakeKlines(_rows_from_df(pd.concat([prefix, existing])))
    out = Downloader(client=client, retries=1, store=store).klines("BTCUSDT", "1m", days=1)

    first_existing = _to_ms(existing.index[0])
    assert client.calls, "префікс має качатися"
    assert all(c < first_existing for c in client.calls)
    assert out.index[0] == prefix.index[0]
    assert out.index[-1] == existing.index[-1]
    # закриті бари середини не змінили close
    assert float(out.loc[existing.index[10], "close"]) == float(existing.iloc[10]["close"])
    assert len(store.load_klines("BTCUSDT", "1m")) == len(out)


def test_klines_fills_internal_gap(monkeypatch) -> None:
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    left = _bars_df("2024-01-09 12:00", 3 * 60)  # до 14:59
    gap = _bars_df("2024-01-09 15:00", 60)  # 15:00..15:59
    right = _bars_df("2024-01-09 16:00", 20 * 60)  # 16:00..11:59
    existing = pd.concat([left, right])
    store = _MemStore()
    store.save_klines("BTCUSDT", "1m", existing)
    client = _FakeKlines(_rows_from_df(pd.concat([left, gap, right])))
    out = Downloader(client=client, retries=1, store=store).klines("BTCUSDT", "1m", days=1)

    gap_start = _to_ms(gap.index[0])
    gap_end = _to_ms(right.index[0])
    assert client.calls
    assert all(gap_start <= c < gap_end for c in client.calls)
    assert gap.index[0] in out.index
    assert len(out) == 24 * 60


def test_klines_fresh_complete_cache_skips_network(monkeypatch) -> None:
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    existing = _bars_df("2024-01-09 12:00", 24 * 60)
    store = _MemStore()
    store.save_klines("BTCUSDT", "1m", existing)
    client = _FakeKlines(_rows_from_df(existing))
    out = Downloader(client=client, retries=1, store=store).klines("BTCUSDT", "1m", days=1)
    assert client.calls == []
    assert len(out) == len(existing)


def test_interval_ms_1m() -> None:
    assert _interval_ms("1m") == 60_000
