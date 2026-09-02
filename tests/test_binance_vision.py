"""Vision-дампи: пропускати календарні дні, які вже повні в кеші."""

from __future__ import annotations

from datetime import date

import pandas as pd
from scalper_hft.data.binance_vision import complete_trade_days, download_agg_trades_vision, missing_vision_periods


def _trades_hours(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="1h")
    return pd.DataFrame(
        {"trade_id": range(len(idx)), "price": 1.0, "amount": 1.0, "side": "buy"},
        index=idx,
    )


def test_complete_trade_days_requires_long_span() -> None:
    short = _trades_hours("2026-08-01 00:00", "2026-08-01 05:00")
    assert complete_trade_days(short, date(2026, 8, 5)) == set()
    full = _trades_hours("2026-08-01 00:00", "2026-08-01 23:00")
    assert complete_trade_days(full, date(2026, 8, 5)) == {date(2026, 8, 1)}
    assert complete_trade_days(full, date(2026, 8, 1)) == set()  # today не вважаємо повним


def test_missing_vision_periods_skips_complete_days() -> None:
    existing = _trades_hours("2026-08-01 00:00", "2026-08-03 23:00")
    missing = missing_vision_periods(
        existing,
        date(2026, 8, 1),
        date(2026, 8, 5),
        "daily",
        date(2026, 8, 5),
    )
    assert date(2026, 8, 1) not in missing
    assert date(2026, 8, 2) not in missing
    assert date(2026, 8, 3) not in missing
    assert date(2026, 8, 4) in missing
    assert date(2026, 8, 5) in missing


def test_missing_vision_periods_monthly_skips_full_month() -> None:
    existing = _trades_hours("2026-07-01 00:00", "2026-07-31 23:00")
    missing = missing_vision_periods(
        existing,
        date(2026, 7, 1),
        date(2026, 7, 31),
        "monthly",
        date(2026, 8, 5),
    )
    assert missing == []


def test_vision_download_skips_http_when_days_cached(monkeypatch) -> None:
    existing = _trades_hours("2026-08-01 00:00", "2026-08-02 23:00")

    class _Store:
        def load_trades(self, symbol: str) -> pd.DataFrame:
            return existing

        def save_trades(self, symbol: str, df: pd.DataFrame) -> None:
            raise AssertionError("не зберігаємо, якщо нічого не докачали")

    calls: list[str] = []

    def _fake_zip(url: str, retries: int = 3, timeout: int = 120) -> bytes | None:
        calls.append(url)
        raise AssertionError(f"не має качати {url}")

    monkeypatch.setattr("scalper_hft.data.store.get_store", lambda: _Store())
    monkeypatch.setattr("scalper_hft.data.binance_vision._download_zip", _fake_zip)
    out = download_agg_trades_vision("BTCUSDT", start=date(2026, 8, 1), end=date(2026, 8, 2))
    assert calls == []
    assert len(out) == len(existing)
