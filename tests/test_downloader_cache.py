"""Інкрементальний кеш klines: префікс, дірки, хвіст; без перекачування середини."""

from __future__ import annotations

import pandas as pd
import pytest
from scalper_hft.data.downloader import (
    Downloader,
    _interval_ms,
    _to_ms,
    agg_trades_cache_fresh,
    format_ms_windows,
    klines_coverage,
    merge_windows,
    missing_klines_windows,
)


@pytest.fixture(autouse=True)
def _isolate_binance_rest_files(tmp_path, monkeypatch) -> None:
    """Cooldown/lock-файли не повинні чіпати ./data і не блокувати pytest."""
    from scalper_hft.data import downloader as dl

    monkeypatch.setattr(dl, "_cooldown_path", lambda: tmp_path / ".binance_rest_cooldown")
    monkeypatch.setattr(
        dl,
        "_exclusive_fetch",
        _noop_exclusive_fetch,
    )


def _noop_exclusive_fetch(_name: str):
    from contextlib import contextmanager

    @contextmanager
    def _inner():
        yield

    return _inner()


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
        self.funding: dict[str, pd.DataFrame] = {}
        self.trades: dict[str, pd.DataFrame] = {}

    def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
        df = self.data.get((symbol, interval))
        return None if df is None else df.copy()

    def save_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        self.data[(symbol, interval)] = df.copy()

    def load_funding(self, symbol: str) -> pd.DataFrame | None:
        df = self.funding.get(symbol)
        return None if df is None else df.copy()

    def save_funding(self, symbol: str, df: pd.DataFrame) -> None:
        self.funding[symbol] = df.copy()

    def load_trades(self, symbol: str) -> pd.DataFrame | None:
        df = self.trades.get(symbol)
        return None if df is None else df.copy()

    def save_trades(self, symbol: str, df: pd.DataFrame) -> None:
        self.trades[symbol] = df.copy()


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


def test_format_ms_windows_empty() -> None:
    assert format_ms_windows([]) == "—"


def test_klines_coverage_empty_is_full_range() -> None:
    now = pd.Timestamp("2024-01-10 12:00:00")
    cov = klines_coverage(None, days=1, interval="1m", now=now)
    assert not cov.complete
    assert cov.n_rows == 0
    text = cov.summary("BTCUSDT", "1m")
    assert "кеш порожній" in text
    assert "2024-01-09 12:00:00" in text


def test_klines_coverage_complete() -> None:
    now = pd.Timestamp("2024-01-10 12:00:00")
    existing = _bars_df("2024-01-09 12:00", 24 * 60)
    cov = klines_coverage(existing, days=1, interval="1m", now=now)
    assert cov.complete
    assert cov.n_rows == 24 * 60
    assert "нічого докачувати" in cov.summary("BTCUSDT", "1m")


def test_klines_coverage_prefix_lists_missing_window() -> None:
    now = pd.Timestamp("2024-01-10 12:00:00")
    existing = _bars_df("2024-01-09 18:00", 18 * 60)
    cov = klines_coverage(existing, days=1, interval="1m", now=now)
    assert not cov.complete
    text = cov.summary("ETHUSDT", "1m")
    assert "бракує 1 вікно" in text
    assert "2024-01-09 12:00:00" in text
    assert "2024-01-09 18:00:00" in text


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


def test_calculate_backoff_ip_ban() -> None:
    from scalper_hft.data.downloader import _calculate_backoff

    exc = RuntimeError("binance 418 IP banned until further notice")
    sleep_s, cat = _calculate_backoff(attempt=0, exc=exc, max_retries=8)
    assert cat == "ip_ban"
    assert 60.0 <= sleep_s <= 66.0

    exc_1003 = RuntimeError("code -1003: Too many requests, IP has been auto-banned")
    sleep_s2, cat2 = _calculate_backoff(attempt=1, exc=exc_1003, max_retries=8)
    assert cat2 == "ip_ban"
    assert 120.0 <= sleep_s2 <= 126.0


def test_calculate_backoff_ip_ban_with_timestamp(monkeypatch) -> None:
    import time

    from scalper_hft.data.downloader import _calculate_backoff

    now = 1700000000.0
    ban_until_ms = int((now + 45.0) * 1000)
    monkeypatch.setattr(time, "time", lambda: now)

    exc = RuntimeError(f"binance 418 IP banned until {ban_until_ms}")
    sleep_s, cat = _calculate_backoff(attempt=0, exc=exc, max_retries=8)
    assert cat == "ip_ban"
    assert abs(sleep_s - 47.0) < 0.5


def test_calculate_backoff_rate_limit() -> None:
    from scalper_hft.data.downloader import _calculate_backoff

    exc = RuntimeError("binance 429 Too many requests")
    sleep_s, cat = _calculate_backoff(attempt=0, exc=exc, max_retries=8)
    assert cat == "rate_limit"
    assert 10.0 <= sleep_s <= 16.0

    sleep_s2, cat2 = _calculate_backoff(attempt=2, exc=exc, max_retries=8)
    assert cat2 == "rate_limit"
    assert 40.0 <= sleep_s2 <= 46.0


def test_calculate_backoff_429_with_code_1003_is_rate_limit_not_ip_ban() -> None:
    """Binance 429 {"code":-1003} — ліміт 6000 req/min, не HTTP 418 ban."""
    from scalper_hft.data.downloader import _calculate_backoff

    exc = RuntimeError(
        'binance 429 Too Many Requests {"code":-1003,"msg":"Too many requests; '
        "current limit of IP(188.163.81.59) is 6000 requests per minute. "
        'Please use the websocket for live updates to avoid polling the API."}'
    )
    sleep_s, cat = _calculate_backoff(attempt=0, exc=exc, max_retries=8)
    assert cat == "rate_limit"
    assert 10.0 <= sleep_s <= 16.0


def test_calculate_backoff_network_error() -> None:
    from scalper_hft.data.downloader import _calculate_backoff

    exc = RuntimeError("Connection reset by peer / 502 Bad Gateway")
    sleep_s, cat = _calculate_backoff(attempt=0, exc=exc, max_retries=8)
    assert cat == "network"
    assert 2.0 <= sleep_s <= 5.0


def test_downloader_checkpointing_saves_intermediate(monkeypatch) -> None:
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)

    # 3 батчі по 1000 свічок (всього 3000 свічок)
    bars = _bars_df("2024-01-08 00:00", 3000)
    store = _MemStore()

    class _MockClient:
        def __init__(self) -> None:
            self.rows = _rows_from_df(bars)

        def fetch_klines(self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000):
            return [r for r in self.rows if r[0] >= since_ms][:limit]

    # чекпоінт кожні 2 батчі
    dl_inst = Downloader(
        client=_MockClient(),
        retries=1,
        store=store,
        batch_delay=0.0,
        checkpoint_batches=2,
    )
    out = dl_inst.klines("BTCUSDT", "1m", days=3)
    assert len(out) == 3000
    saved = store.load_klines("BTCUSDT", "1m")
    assert saved is not None and len(saved) == 3000


def test_downloader_saves_partial_progress_on_failure(monkeypatch) -> None:
    import pytest
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)

    bars = _bars_df("2024-01-08 00:00", 3000)
    store = _MemStore()

    class _FailingClient:
        def __init__(self) -> None:
            self.rows = _rows_from_df(bars)
            self.calls = 0

        def fetch_klines(self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000):
            self.calls += 1
            if self.calls == 2:
                # Падаємо на другому батчі після успішного першого
                raise RuntimeError("Simulated network fatal failure")
            return [r for r in self.rows if r[0] >= since_ms][:limit]

    # retries=1 щоб швидше впасти
    dl_inst = Downloader(
        client=_FailingClient(),
        retries=1,
        store=store,
        batch_delay=0.0,
    )

    with pytest.raises(RuntimeError, match="Не вдалося завантажити дані"):
        dl_inst.klines("BTCUSDT", "1m", days=3)

    # Перевіряємо, що перший батч (1000 свічок) зберігся у сховищі незважаючи на падіння!
    saved = store.load_klines("BTCUSDT", "1m")
    assert saved is not None
    assert len(saved) == 1000


def test_download_klines_skips_downloader_when_cache_complete(monkeypatch) -> None:
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    existing = _bars_df("2024-01-09 12:00", 24 * 60)

    class _Store:
        def load_klines(self, symbol: str, interval: str) -> pd.DataFrame:
            return existing

        def save_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
            raise AssertionError("повний кеш не перезаписується")

    monkeypatch.setattr(dl, "get_store", lambda: _Store())

    class _Boom:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("Downloader не має створюватися")

    monkeypatch.setattr(dl, "Downloader", _Boom)
    out = dl.download_klines("BTCUSDT", "1m", days=1)
    assert len(out) == len(existing)


def test_funding_fetches_only_from_last_when_history_covers(monkeypatch) -> None:
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    existing = pd.DataFrame(
        {"fundingRate": [0.0001] * 6},
        index=pd.date_range("2024-01-08 00:00", periods=6, freq="8h"),
    )
    store = _MemStore()
    store.save_funding("BTCUSDT", existing)

    class _Client:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def fetch_funding_rate_history(self, symbol: str, since_ms: int, limit: int = 1000) -> list[dict]:
            self.calls.append(int(since_ms))
            return []

    client = _Client()
    Downloader(client=client, retries=1, store=store, batch_delay=0.0).funding("BTCUSDT", days=2)
    assert client.calls
    last_ms = _to_ms(existing.index[-1])
    assert all(c >= last_ms for c in client.calls)
    needed_from = _to_ms(now - pd.Timedelta(days=2))
    assert client.calls[0] > needed_from


def test_agg_trades_continues_from_last_cached_trade(monkeypatch) -> None:
    """Докачка НЕ перекачує кеш: стартує від останньої закешованої угоди.

    Після аудиту 2026-09-11 (знахідка K5) продовження йде за `fromId`, а не за
    часом: часовий курсор `last_ts + 1` перестрибував цілу мілісекунду й губив
    угоди, що ділять її з останньою закешованою. Тут перевіряються ОБИДВІ гілки:
    клієнт без підтримки from_id (фолбек за часом) і з підтримкою.
    """
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    last = pd.Timestamp("2024-01-10 10:00:00")
    existing = pd.DataFrame(
        {"trade_id": [1], "price": [100.0], "amount": [0.1], "side": ["buy"]},
        index=[last],
    )
    store = _MemStore()
    store.save_trades("BTCUSDT", existing)

    class _ClientWithoutFromId:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def fetch_agg_trades(self, symbol: str, since_ms: int, limit: int = 1000) -> list[dict]:
            self.calls.append(int(since_ms))
            return []

    client = _ClientWithoutFromId()
    Downloader(client=client, retries=1, store=store, batch_delay=0.0).agg_trades("BTCUSDT", days=2)
    assert client.calls
    assert client.calls[0] >= _to_ms(last)

    class _ClientWithFromId:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def fetch_agg_trades(
            self, symbol: str, since_ms: int | None = None, limit: int = 1000, *, from_id: int | None = None
        ) -> list[dict]:
            self.calls.append((since_ms, from_id))
            return []

    client2 = _ClientWithFromId()
    Downloader(client=client2, retries=1, store=store, batch_delay=0.0).agg_trades("BTCUSDT", days=2)
    assert client2.calls
    # Кеш містить реальний trade_id=1 → продовжуємо з 2 (fromId ІНКЛЮЗИВНО),
    # без startTime (Binance забороняє передавати їх разом).
    assert client2.calls[0] == (None, 2)


def test_agg_trades_cache_fresh() -> None:
    now = pd.Timestamp("2024-01-10 12:00:00")
    assert not agg_trades_cache_fresh(None, now=now)
    assert not agg_trades_cache_fresh(pd.DataFrame(), now=now)
    stale = pd.DataFrame(
        {"trade_id": [1], "price": [1.0], "amount": [1.0], "side": ["buy"]},
        index=[now - pd.Timedelta(days=3)],
    )
    fresh = pd.DataFrame(
        {"trade_id": [1], "price": [1.0], "amount": [1.0], "side": ["buy"]},
        index=[now - pd.Timedelta(hours=6)],
    )
    assert not agg_trades_cache_fresh(stale, now=now)
    assert agg_trades_cache_fresh(fresh, now=now)


def test_download_agg_trades_fresh_cache_skips_network(monkeypatch) -> None:
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    existing = pd.DataFrame(
        {"trade_id": [1], "price": [0.1], "amount": [1.0], "side": ["buy"]},
        index=[now - pd.Timedelta(hours=1)],
    )

    class _Store:
        def load_trades(self, symbol: str) -> pd.DataFrame:
            return existing

        def save_trades(self, symbol: str, df: pd.DataFrame) -> None:
            raise AssertionError("свіжий кеш не перезаписується")

    monkeypatch.setattr(dl, "get_store", lambda: _Store())

    class _Boom:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("Downloader не має створюватися")

    monkeypatch.setattr(dl, "Downloader", _Boom)
    out = dl.download_agg_trades("DOGEUSDT", days=60)
    assert len(out) == 1


def test_download_agg_trades_rechecks_cache_after_lock(monkeypatch) -> None:
    """Під lock інший воркер уже зберіг кеш — REST не викликається."""
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    filled = pd.DataFrame(
        {"trade_id": [1], "price": [0.1], "amount": [1.0], "side": ["buy"]},
        index=[now - pd.Timedelta(minutes=5)],
    )
    n_load = {"n": 0}

    class _Store:
        def load_trades(self, symbol: str) -> pd.DataFrame | None:
            n_load["n"] += 1
            if n_load["n"] == 1:
                return None
            return filled

        def save_trades(self, symbol: str, df: pd.DataFrame) -> None:
            raise AssertionError("не має зберігати")

    monkeypatch.setattr(dl, "get_store", lambda: _Store())

    class _Boom:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("Downloader не має створюватися після lock")

    monkeypatch.setattr(dl, "Downloader", _Boom)
    out = dl.download_agg_trades("DOGEUSDT", days=60)
    assert len(out) == 1
    assert n_load["n"] == 2


def test_set_shared_cooldown_writes_future_ts(tmp_path, monkeypatch) -> None:
    import time

    from scalper_hft.data import downloader as dl

    path = tmp_path / ".binance_rest_cooldown"
    monkeypatch.setattr(dl, "_cooldown_path", lambda: path)
    dl._set_shared_cooldown(30.0)
    until = float(path.read_text(encoding="utf-8").strip())
    assert until > time.time() + 20
    dl._set_shared_cooldown(5.0)
    until2 = float(path.read_text(encoding="utf-8").strip())
    assert until2 == until
    path.write_text(str(time.time() - 10.0), encoding="utf-8")
    dl._wait_shared_cooldown()  # минуле — без sleep


def _good_klines_df(n: int, start: str = "2024-01-01", freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    close = pd.Series(range(n), index=idx, dtype=float) + 100.0
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0},
        index=idx,
    )


def _assert_valid_klines(df: pd.DataFrame | None) -> None:
    if df is None:
        return
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns
    assert len(df) > 0
    assert df.index.is_monotonic_increasing


def test_parquet_reader_never_sees_partial_file(tmp_path) -> None:
    """Concurrent writer + reader: old/new complete snapshots or None, never corrupt parquet."""
    import threading
    import time

    from scalper_hft.data.storage import load_klines, save_klines

    path = tmp_path / "BTCUSDT_1m_klines.parquet"
    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        for i in range(40):
            df = _good_klines_df(120 + i, start=f"2024-01-{1 + (i % 28):02d}")
            save_klines(path, df)
            time.sleep(0.002)

    def reader() -> None:
        while not stop.is_set():
            try:
                df = load_klines(path)
                _assert_valid_klines(df)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
            time.sleep(0.001)

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start()
    r.start()
    w.join(timeout=30)
    stop.set()
    r.join(timeout=5)
    assert not errors, "reader saw corrupt parquet:\n" + "\n".join(errors[:5])
    final = load_klines(path)
    assert final is not None and len(final) >= 120


def test_two_writers_last_wins_complete(tmp_path) -> None:
    """Two writers with file lock: final parquet is valid and complete."""
    import threading

    from scalper_hft.data.storage import load_klines, save_klines

    path = tmp_path / "ETHUSDT_1m_klines.parquet"
    barrier = threading.Barrier(2)

    def writer(tag: str, n: int) -> None:
        barrier.wait(timeout=5)
        for i in range(15):
            df = _good_klines_df(n + i, start=f"2024-02-{1 + (i % 20):02d}")
            df.attrs["tag"] = tag  # noqa: B003 — not persisted, only to vary frames
            save_klines(path, df)

    t1 = threading.Thread(target=writer, args=("a", 100))
    t2 = threading.Thread(target=writer, args=("b", 200))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    loaded = load_klines(path)
    assert loaded is not None
    _assert_valid_klines(loaded)
    assert len(loaded) >= 100


def test_with_retry_sets_shared_cooldown_on_429(tmp_path, monkeypatch) -> None:
    from scalper_hft.data import downloader as dl

    path = tmp_path / ".binance_rest_cooldown"
    monkeypatch.setattr(dl, "_cooldown_path", lambda: path)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)
    sleeps: list[float] = []
    monkeypatch.setattr(dl, "_set_shared_cooldown", lambda s: sleeps.append(s))

    class _Once429:
        def __init__(self) -> None:
            self.n = 0

        def fetch_klines(self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000):
            self.n += 1
            if self.n == 1:
                raise RuntimeError('binance 429 Too Many Requests {"code":-1003,"msg":"Too many requests"}')
            return []

    store = _MemStore()
    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    Downloader(client=_Once429(), retries=3, store=store, batch_delay=0.0).klines("BTCUSDT", "1m", days=1)
    assert sleeps
    assert sleeps[0] >= 10.0


def test_downloader_describe_source() -> None:
    dl_inst = Downloader(exchange_id="binanceusdm", client=_FakeKlines([]))
    desc = dl_inst.describe_source()
    assert "binanceusdm" in desc
    assert "LIVE (НЕ testnet ✓)" in desc


def test_downloader_logs_source_on_klines_download(monkeypatch, caplog) -> None:
    import logging
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    prefix = _bars_df("2024-01-09 12:00", 24 * 60)
    client = _FakeKlines(_rows_from_df(prefix))
    store = _MemStore()

    with caplog.at_level(logging.INFO):
        Downloader(client=client, retries=1, store=store, exchange_id="binanceusdm").klines("BTCUSDT", "1m", days=1)

    assert any(
        "Докачую klines BTCUSDT 1m з джерела" in record.message and "LIVE (НЕ testnet" in record.message
        for record in caplog.records
    )


def test_downloader_logs_source_on_agg_trades_download(monkeypatch, caplog) -> None:
    import logging
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    store = _MemStore()

    class _FakeTrades:
        def fetch_agg_trades(self, symbol: str, since: int, limit: int = 1000):
            return []

    with caplog.at_level(logging.INFO):
        Downloader(client=_FakeTrades(), retries=1, store=store, exchange_id="binanceusdm").agg_trades("BTCUSDT", days=1)

    assert any(
        "aggTrades BTCUSDT: докачую з REST" in record.message and "LIVE (НЕ testnet" in record.message
        for record in caplog.records
    )


def test_downloader_logs_source_on_funding_download(monkeypatch, caplog) -> None:
    import logging
    from scalper_hft.data import downloader as dl

    now = pd.Timestamp("2024-01-10 12:00:00")
    monkeypatch.setattr(dl, "_utc_now", lambda: now)
    store = _MemStore()

    class _FakeFunding:
        def fetch_funding_rate_history(self, symbol: str, since: int, limit: int = 1000):
            return [{"timestamp": since, "fundingRate": 0.0001}]

    with caplog.at_level(logging.INFO):
        Downloader(
            client=_FakeFunding(),
            retries=1,
            store=store,
            exchange_id="binanceusdm",
            strict_funding_coverage=False,
        ).funding("BTCUSDT", days=1)

    assert any(
        "funding BTCUSDT: докачую з REST" in record.message and "LIVE (НЕ testnet" in record.message
        for record in caplog.records
    )

