"""Vision-дампи: пропускати календарні дні, які вже повні в кеші."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
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


def test_complete_trade_days_rejects_time_complete_but_gappy_ids() -> None:
    """День із повним часовим проміжком, але втраченими aggTrade id — НЕ повний.

    Реальний кейс: кеш BTCUSDT покривав 39.3% угод, маючи при цьому повний
    span по кожному дню. Без цієї перевірки vision пропускав би такі дні
    («день виглядає повним») і втрата лишалась би назавжди.
    """
    idx = pd.date_range("2026-08-01 00:00", "2026-08-01 23:00", freq="1h")
    gappy = pd.DataFrame(
        {"trade_id": [i * 10 for i in range(len(idx))], "price": 1.0, "amount": 1.0, "side": "buy"},
        index=idx,
    )
    assert complete_trade_days(gappy, date(2026, 8, 5)) == set()

    dense = pd.DataFrame(
        {"trade_id": range(len(idx)), "price": 1.0, "amount": 1.0, "side": "buy"},
        index=idx,
    )
    assert complete_trade_days(dense, date(2026, 8, 5)) == {date(2026, 8, 1)}


def test_missing_vision_periods_refetches_gappy_days() -> None:
    idx = pd.date_range("2026-08-01 00:00", "2026-08-03 23:00", freq="1h")
    gappy = pd.DataFrame(
        {"trade_id": [i * 10 for i in range(len(idx))], "price": 1.0, "amount": 1.0, "side": "buy"},
        index=idx,
    )
    missing = missing_vision_periods(gappy, date(2026, 8, 1), date(2026, 8, 5), "daily", date(2026, 8, 5))
    assert date(2026, 8, 1) in missing
    assert date(2026, 8, 2) in missing
    assert date(2026, 8, 3) in missing


def test_complete_trade_days_synthetic_ids_are_not_refetched_forever() -> None:
    """Без реальних id оцінити повноту неможливо → день лишається «повним».

    Інакше цикл докачки ніколи не завершився б: синтетичні id завжди
    виглядали б як «непокриті».
    """
    idx = pd.date_range("2026-08-01 00:00", "2026-08-01 23:00", freq="1h")
    synth = pd.DataFrame(
        {"trade_id": [-i - 1 for i in range(len(idx))], "price": 1.0, "amount": 1.0, "side": "buy"},
        index=idx,
    )
    assert complete_trade_days(synth, date(2026, 8, 5)) == {date(2026, 8, 1)}


# ── Checkpoint / resume ─────────────────────────────────────────────────────


def _zip_bytes(day: str, *, start_id: int | None = None, n: int = 5) -> bytes:
    """Синтетичний vision-архів (CSV без заголовка, як у Binance).

    `trade_id` унікальні ДЛЯ КОЖНОГО ДНЯ (інакше дедуп за trade_id злив би дні).
    """
    import io
    import zipfile

    base = int(pd.Timestamp(f"{day} 00:00", tz="UTC").timestamp() * 1000)
    if start_id is None:
        start_id = int(day.replace("-", "")) * 10
    # рядки розкидані на ~24 год: інакше день не вважається "повним" (span ≥ 20 год)
    offsets = [0, 6 * 3600_000, 12 * 3600_000, 18 * 3600_000, 23 * 3600_000 + 59 * 60_000]
    rows = "\n".join(
        f"{start_id + i},{100.0 + i},1.0,{i},{i},{base + offsets[i % len(offsets)]},{'true' if i % 2 else 'false'}"
        for i in range(n)
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"SYN-aggTrades-{day}.csv", rows)
    return buf.getvalue()


class _MemStore:
    def __init__(self, initial: pd.DataFrame | None = None) -> None:
        self.df = initial
        self.saves = 0

    def load_trades(self, symbol: str):
        return self.df

    def save_trades(self, symbol: str, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.saves += 1


def test_vision_checkpoints_every_n_archives(monkeypatch) -> None:
    """Дані мають потрапляти на диск ДО кінця циклу."""
    store = _MemStore()
    monkeypatch.setattr("scalper_hft.data.store.get_store", lambda: store)
    monkeypatch.setattr("scalper_hft.data.binance_vision.time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "scalper_hft.data.binance_vision._download_zip",
        lambda url, **kw: _zip_bytes(url[-14:-4]),
    )
    out = download_agg_trades_vision(
        "SYNUSDT", start=date(2026, 8, 10), end=date(2026, 8, 14), checkpoint_every=2
    )
    # 5 архівів (10…14 серпня) × 5 рядків; checkpoint після 2-го і 4-го + фінальний
    assert store.saves == 3
    assert len(out) == 25


def test_vision_resumes_after_interruption(monkeypatch) -> None:
    """Обрив посеред циклу не втрачає вже завантажене — повторний запуск докачує решту.

    До виправлення дані тримались у пам'яті до кінця і при обриві втрачались усі.
    """
    store = _MemStore()
    monkeypatch.setattr("scalper_hft.data.store.get_store", lambda: store)
    monkeypatch.setattr("scalper_hft.data.binance_vision.time.sleep", lambda _s: None)

    fetched: list[str] = []

    def _zip_but_die_on_third(url: str, **kw):
        day = url[-14:-4]
        fetched.append(day)
        if day == "2026-08-12":
            raise KeyboardInterrupt("обрив на третьому архіві")
        return _zip_bytes(day)

    monkeypatch.setattr("scalper_hft.data.binance_vision._download_zip", _zip_but_die_on_third)

    with pytest.raises(KeyboardInterrupt):
        download_agg_trades_vision("SYNUSDT", start=date(2026, 8, 10), end=date(2026, 8, 14), checkpoint_every=1)

    assert store.saves == 2, "перші два архіви мали бути збережені до обриву"
    assert store.df is not None and len(store.df) == 10

    # ── повторний запуск: дні 10 і 11 уже в кеші, качаємо лише решту
    fetched.clear()
    monkeypatch.setattr(
        "scalper_hft.data.binance_vision._download_zip",
        lambda url, **kw: (fetched.append(url[-14:-4]) or _zip_bytes(url[-14:-4])),
    )
    out = download_agg_trades_vision("SYNUSDT", start=date(2026, 8, 10), end=date(2026, 8, 14))

    assert "2026-08-10" not in fetched, "уже повний день не перекачується"
    assert "2026-08-11" not in fetched
    assert set(fetched) == {"2026-08-12", "2026-08-13", "2026-08-14"}
    assert len(out) == 25, "10 старих + 15 нових"


def test_vision_no_checkpoint_mode_still_works(monkeypatch) -> None:
    """checkpoint_every=0 — стара поведінка (один запис у кінці)."""
    store = _MemStore()
    monkeypatch.setattr("scalper_hft.data.store.get_store", lambda: store)
    monkeypatch.setattr("scalper_hft.data.binance_vision.time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "scalper_hft.data.binance_vision._download_zip",
        lambda url, **kw: _zip_bytes(url[-14:-4]),
    )
    out = download_agg_trades_vision("SYNUSDT", start=date(2026, 8, 10), end=date(2026, 8, 12), checkpoint_every=0)
    assert store.saves == 1
    assert len(out) == 15


def test_cli_download_passes_vision_checkpoint_flag(monkeypatch) -> None:
    """`--vision-checkpoint-every` має доходити до download_agg_trades_vision.

    Без цього в режимі `monthly` типове вікно чекпоінта (5 архівів) означало б,
    що обрив коштує до п'яти місяців завантаження.
    """
    import argparse

    from scalper_hft.cli import main as cli_main
    from scalper_hft.cli import ops as cli_ops

    seen: dict = {}

    def _fake_vision(symbol, start=None, end=None, freq="daily", *, checkpoint_every=5):
        seen.update(symbol=symbol, freq=freq, checkpoint_every=checkpoint_every)
        return pd.DataFrame()

    monkeypatch.setattr("scalper_hft.data.binance_vision.download_agg_trades_vision", _fake_vision)
    monkeypatch.setattr(cli_ops, "download_klines", lambda *a, **k: pd.DataFrame(), raising=False)

    args = argparse.Namespace(
        symbol="SOLUSDT",
        interval="1h",
        days=1,
        exchange=None,
        trades=False,
        trades_days=None,
        funding=False,
        force=False,
        retries=1,
        delay=0.0,
        checkpoint_batches=0,
        vision=True,
        vision_start="2026-03-16",
        vision_freq="monthly",
        vision_checkpoint_every=1,
    )
    cli_ops.cmd_download(args)

    assert seen == {"symbol": "SOLUSDT", "freq": "monthly", "checkpoint_every": 1}
    assert cli_main is not None  # модуль імпортується без циклів
