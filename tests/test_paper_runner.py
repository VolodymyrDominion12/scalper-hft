"""Тести PaperRunner (без мережі та sleep)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.paper_runner import PaperRunner, PaperRunResult, _fetch_recent


class _FlatStrategy:
    name = "flat"
    param_space: dict = {}
    needs_trades = False
    needs_funding = False

    def __init__(self, **kwargs) -> None:
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index)


def _sample_df(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = pd.Series(100.0, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=idx,
    )


def test_paper_run_result_summary_empty() -> None:
    result = PaperRunResult()
    assert "жодного кроку" in result.summary()


def test_paper_runner_step_uses_closed_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_df()
    monkeypatch.setattr("scalper_hft.live.paper_runner._fetch_recent", lambda *a, **k: df)

    runner = PaperRunner(_FlatStrategy(), "BTCUSDT", "1m")
    action = runner.step()
    assert isinstance(action, str)
    assert action.startswith("hold") or "error" not in action


def test_paper_runner_run_no_sleep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_df()
    monkeypatch.setattr("scalper_hft.live.paper_runner._fetch_recent", lambda *a, **k: df)
    monkeypatch.setattr("scalper_hft.live.paper_runner.time.sleep", lambda _s: None)

    runner = PaperRunner(_FlatStrategy(), "BTCUSDT", "1m", account=PaperAccount(10_000.0))
    result = runner.run(iterations=3, sleep_sec=60, out_dir=tmp_path)

    assert len(result.equity_points) == 3
    assert len(result.actions) == 3
    assert (tmp_path / "paper_equity_BTCUSDT.csv").exists()


def test_paper_runner_rejects_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from scalper_hft.config import get_settings, set_settings

    orig = get_settings()
    try:
        set_settings(dataclasses.replace(orig, dry_run=False))
        with pytest.raises(RuntimeError, match="paper-only"):
            PaperRunner(_FlatStrategy(), "BTCUSDT")
    finally:
        set_settings(orig)


def test_fetch_recent_builds_index(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.fetch_klines.return_value = [
        [1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 10.0],
        [1_700_000_060_000, 100.5, 102.0, 100.0, 101.0, 12.0],
    ]
    monkeypatch.setattr("scalper_hft.live.paper_runner.ExchangeClient", lambda: client)
    df = _fetch_recent("BTCUSDT", "1m", limit=2)
    assert len(df) == 2
    assert "close" in df.columns


def test_fetch_recent_asks_for_recent_candles_not_listing_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """`since_ms=0` означав `startTime=0` → Binance віддавав свічки 2019 року.

    Аудит 2026-09-11 (K6): через це paper-демони рахували сигнал за 2019 рік і
    «виконували» його за ціною 2019 року, а equity не рухалась.
    """
    client = MagicMock()
    client.fetch_klines.return_value = [[1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 10.0]]
    monkeypatch.setattr("scalper_hft.live.paper_runner.ExchangeClient", lambda: client)

    _fetch_recent("BTCUSDT", "1h", limit=600)

    # Перша сторінка — не остання: пагінація добирає вікно до поточного моменту.
    since_ms = client.fetch_klines.call_args_list[0].kwargs["since_ms"]
    assert since_ms > 0, "нуль — це startTime=0, тобто найстаріші свічки лістингу"
    age_hours = (pd.Timestamp.now("UTC").tz_localize(None).timestamp() * 1000 - since_ms) / 3_600_000
    # 600 годинних барів + запас ×2 → трохи більше 1200 годин.
    assert 1200 <= age_hours <= 1300, age_hours


def test_recent_since_ms_is_timeframe_aware() -> None:
    from scalper_hft.data.client import recent_since_ms

    now = pd.Timestamp("2026-09-11 12:00:00")
    assert recent_since_ms("1h", 600, now=now) == int((now - pd.Timedelta(hours=1200)).timestamp() * 1000)
    assert recent_since_ms("5m", 600, now=now) == int((now - pd.Timedelta(minutes=6000)).timestamp() * 1000)
    # limit=1 → кілька хвилин тому, а не 1970 рік.
    assert recent_since_ms("1m", 1, now=now) == int((now - pd.Timedelta(minutes=2)).timestamp() * 1000)
    with pytest.raises(ValueError):
        recent_since_ms("1w", 10, now=now)


class _BinanceLikeClient:
    """Мок Binance/ccxt: віддає ПЕРШІ `limit` барів від `startTime`, без майбутніх.

    Саме така поведінка (перші барів вікна + стеля на розмір батчу) робила
    pad-вікно ×2 без tail-слайсу пасткою.
    """

    def __init__(self, interval_ms: int, page_cap: int = 1000) -> None:
        self.interval_ms = interval_ms
        self.page_cap = page_cap
        self.calls: list[tuple[int, int]] = []

    def fetch_klines(self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> list[list[float]]:
        self.calls.append((int(since_ms), int(limit)))
        cap = min(int(limit), self.page_cap)
        now_ms = _now_ms()
        last_bar = (now_ms // self.interval_ms) * self.interval_ms
        rows: list[list[float]] = []
        for i in range(cap):
            ts = int(since_ms) + i * self.interval_ms
            if ts > last_bar:
                break
            rows.append([ts, 100.0, 101.0, 99.0, 100.0, 1.0])
        return rows


def _now_ms() -> float:
    return pd.Timestamp.now("UTC").tz_localize(None).timestamp() * 1000


@pytest.mark.parametrize(
    ("interval", "limit", "interval_ms"),
    [("1h", 600, 3_600_000), ("1d", 600, 86_400_000), ("1h", 800, 3_600_000), ("4h", 600, 14_400_000)],
)
def test_fetch_recent_klines_window_ends_at_now(interval: str, limit: int, interval_ms: int) -> None:
    """Регресія 2026-09-13: боти бачили старшу половину pad-вікна.

    Було: `fetch_klines(since=now−2L, limit=L)` → останній бар за L барів до
    `now` (pairs 1h → 33 доби, ts_momentum 1d → 600 діб). Має бути tail.
    """
    from scalper_hft.data.client import MAX_KLINES_PER_REQUEST, fetch_recent_klines

    client = _BinanceLikeClient(interval_ms)
    batch = fetch_recent_klines(client, "BTCUSDT", interval, limit)

    assert len(batch) == limit
    since_ms, asked = client.calls[-1]
    assert asked <= MAX_KLINES_PER_REQUEST
    # вікно запиту мусить доходити до поточного моменту
    assert since_ms + asked * interval_ms >= _now_ms() - interval_ms
    age_bars = (_now_ms() - batch[-1][0]) / interval_ms
    assert 0 <= age_bars <= 2, f"останній бар застарілий на {age_bars:.1f} барів"


def test_fetch_recent_klines_respects_request_cap() -> None:
    """limit=800, pad=2 → 1600 > ліміту Binance: вікно все одно має доходити до now."""
    from scalper_hft.data.client import MAX_KLINES_PER_REQUEST, fetch_recent_klines

    client = _BinanceLikeClient(3_600_000)
    batch = fetch_recent_klines(client, "LINKUSDT", "1h", 800)

    _, asked = client.calls[-1]
    assert asked == MAX_KLINES_PER_REQUEST
    assert (_now_ms() - batch[-1][0]) / 3_600_000 <= 2


def test_fetch_recent_short_history_returns_all_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Символ, що торгується менше за вікно: повертаємо все, що є, без падіння."""
    from scalper_hft.data.client import fetch_recent_klines

    client = MagicMock()
    client.fetch_klines.return_value = [[1_700_000_000_000 + i * 3_600_000, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(5)]

    batch = fetch_recent_klines(client, "NEWUSDT", "1h", 600)
    assert len(batch) == 5


def test_pairs_fetch_ohlcv_is_not_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """`pairs_runner._fetch_ohlcv` має віддавати свіжі 1h-бари, не 33-добової давнини."""
    from scalper_hft.live.pairs_runner import _fetch_ohlcv

    monkeypatch.setattr(
        "scalper_hft.live.pairs_runner.ExchangeClient",
        lambda *a, **k: _BinanceLikeClient(3_600_000),
    )
    df = _fetch_ohlcv("LINKUSDT", "1h")

    assert len(df) == 800
    age_hours = (_now_ms() - df.index[-1].timestamp() * 1000) / 3_600_000
    assert 0 <= age_hours <= 2, f"останній 1h-бар застарілий на {age_hours:.1f} год"


def test_ts_momentum_fetch_recent_is_not_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """`paper_runner._fetch_recent` для 1d має віддавати свіжі денні бари, не 2025 рік."""
    monkeypatch.setattr(
        "scalper_hft.live.paper_runner.ExchangeClient",
        lambda *a, **k: _BinanceLikeClient(86_400_000),
    )
    df = _fetch_recent("BTCUSDT", "1d")

    assert len(df) == 600
    age_days = (_now_ms() - df.index[-1].timestamp() * 1000) / 86_400_000
    assert 0 <= age_days <= 2, f"останній 1d-бар застарілий на {age_days:.1f} діб"
