"""Тести OOS-дисципліни (Narang гл. 9 — burning data): check_and_burn + wire у audit_cell.

Перевіряє:
- oos_range_from_df коректно дістає діапазон дат;
- check_and_burn дописує використання (ідемпотентно);
- check_and_burn з enforce=True блокує повторне використання того самого вікна;
- audit_cell під OOS_ENFORCE_BURN=true повертає status=error при спаленому вікні;
- audit_cell дописує використання після успішного аудиту (реєстр зростає).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def _klines(n: int = 120, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, n)), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0}, index=idx
    )


def test_oos_range_from_df() -> None:
    from scalper_hft.validation.oos_registry import oos_range_from_df

    df = _klines(48, start="2025-03-10")
    rng = oos_range_from_df(df, days=2)
    assert rng is not None
    assert rng[0] == date(2025, 3, 10)
    assert rng[1] == date(2025, 3, 11)
    assert oos_range_from_df(pd.DataFrame(), days=2) is None


def test_check_and_burn_appends_and_blocks(tmp_path: Path) -> None:
    from scalper_hft.validation.oos_registry import check_and_burn

    reg = tmp_path / "oos.md"
    df = _klines(48, start="2025-04-01")
    ok, reason = check_and_burn(
        strategy="pairs_arb", symbol="BTCUSDT", df=df, days=2, purpose="test", registry_path=reg, enforce=True
    )
    assert ok is True
    assert reason == ""
    # повторне використання того самого вікна — заблоковано
    ok2, reason2 = check_and_burn(
        strategy="pairs_arb", symbol="BTCUSDT", df=df, days=2, purpose="test", registry_path=reg, enforce=True
    )
    assert ok2 is False
    assert "спалене" in reason2


def test_check_and_burn_distinct_windows_ok(tmp_path: Path) -> None:
    from scalper_hft.validation.oos_registry import check_and_burn

    reg = tmp_path / "oos.md"
    df1 = _klines(48, start="2025-04-01")
    df2 = _klines(48, start="2025-05-01")  # інший календарний відрізок
    ok1, _ = check_and_burn(strategy="s", symbol="X", df=df1, days=2, purpose="t", registry_path=reg, enforce=True)
    ok2, _ = check_and_burn(strategy="s", symbol="X", df=df2, days=2, purpose="t", registry_path=reg, enforce=True)
    assert ok1 and ok2


def test_check_and_burn_no_enforce_always_ok(tmp_path: Path) -> None:
    from scalper_hft.validation.oos_registry import check_and_burn

    reg = tmp_path / "oos.md"
    df = _klines(48, start="2025-04-01")
    check_and_burn(strategy="s", symbol="X", df=df, days=2, purpose="t", registry_path=reg, enforce=False)
    # enforce=False → повторне використання не блокується (лише дописується)
    ok, _ = check_and_burn(strategy="s", symbol="X", df=df, days=2, purpose="t", registry_path=reg, enforce=False)
    assert ok is True


def test_check_and_burn_empty_path_disabled() -> None:
    """Path('') == Path('.'): порожній OOS_REGISTRY_PATH не має валити
    read_text('.') з IsADirectoryError — реєстр просто вимкнено."""
    from scalper_hft.validation.oos_registry import check_and_burn

    df = _klines(48, start="2025-06-01")
    for disabled in (Path(""), Path(".")):
        ok, reason = check_and_burn(
            strategy="s", symbol="X", df=df, days=2, purpose="t", registry_path=disabled, enforce=True
        )
        assert ok is True
        assert reason == ""


def test_audit_cell_burns_and_blocks_on_enforce(monkeypatch, tmp_path: Path) -> None:
    """audit_cell під OOS_ENFORCE_BURN=true: перший виклик OK + спалює; другий — error."""
    from scalper_hft import config as cfg
    from scalper_hft.data import access as acc
    from scalper_hft.data import downloader as dl
    from scalper_hft.validation.cell_audit import audit_cell

    reg = tmp_path / "oos.md"
    df = _klines(120, start="2025-01-01")

    class _MemStore:
        def __init__(self) -> None:
            self.data: dict[tuple[str, str], pd.DataFrame] = {}

        def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
            d = self.data.get((symbol, interval))
            return None if d is None else d.copy()

        def save_klines(self, symbol: str, interval: str, d: pd.DataFrame) -> None:
            self.data[(symbol, interval)] = d.copy()

    store = _MemStore()
    store.save_klines("TESTOOSUSDT", "1h", df)
    monkeypatch.setattr(acc, "get_store", lambda: store)
    monkeypatch.setattr(dl, "get_store", lambda: store)
    monkeypatch.setattr(dl, "_utc_now", lambda: pd.Timestamp("2025-01-06 00:00:00"))
    # уникаємо мережі: ensure_klines повертає кеш напряму
    monkeypatch.setattr(acc, "ensure_klines", lambda symbol, interval, days, **kw: df)

    import dataclasses as _dc

    settings = _dc.replace(cfg.get_settings(), enforce_oos_burn=True, oos_registry_path=reg)
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    a1 = audit_cell("mean_reversion", "TESTOOSUSDT", "1h", days=5)
    # перший виклик спалює вікно; статус ok/error від даних, але не від burn
    assert "спалене" not in (a1.error or "")
    a2 = audit_cell("mean_reversion", "TESTOOSUSDT", "1h", days=5)
    # другий виклик — заблокований burn-гейтом
    assert a2.status == "error"
    assert "спалене" in a2.error
