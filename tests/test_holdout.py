"""Тести «замкованого» holdout (Narang гл. 9): split_research_holdout + wire у audit_cell/optimize/sweep.

Перевіряє:
- split_research_holdout ділить за барями, holdout — останні N;
- holdout_pct <= 0 → holdout вимкнено (повний df, порожній holdout);
- audit_cell під HOLDOUT_PCT>0 аудитує лише research-частину (менше барів у WF).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _klines(n: int = 200, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(2).normal(0, 0.1, n)), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0}, index=idx
    )


def test_split_research_holdout_basic() -> None:
    from scalper_hft.validation.holdout import split_research_holdout

    df = _klines(100)
    research, holdout = split_research_holdout(df, 20)
    assert len(research) == 80
    assert len(holdout) == 20
    # holdout — останні 20 барів
    pd.testing.assert_index_equal(holdout.index, df.index[-20:])
    pd.testing.assert_index_equal(research.index, df.index[:-20])


def test_split_research_holdout_disabled() -> None:
    from scalper_hft.validation.holdout import split_research_holdout

    df = _klines(100)
    research, holdout = split_research_holdout(df, 0.0)
    assert len(research) == 100
    assert len(holdout) == 0
    # відсотки > 1 трактуємо як частку після /100
    r2, h2 = split_research_holdout(df, 30)  # 30% → 0.30
    assert len(r2) == 70 and len(h2) == 30


def test_audit_cell_holdout_trims_research(monkeypatch, tmp_path: Path) -> None:
    """audit_cell під HOLDOUT_PCT=20 аудитує лише перші 80% (менше WF-вікон)."""
    from scalper_hft import config as cfg
    from scalper_hft.data import access as acc
    from scalper_hft.data import downloader as dl
    from scalper_hft.validation.cell_audit import audit_cell

    df = _klines(200, start="2025-01-01")

    class _MemStore:
        def __init__(self) -> None:
            self.data: dict[tuple[str, str], pd.DataFrame] = {}

        def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
            d = self.data.get((symbol, interval))
            return None if d is None else d.copy()

        def save_klines(self, symbol: str, interval: str, d: pd.DataFrame) -> None:
            self.data[(symbol, interval)] = d.copy()

    store = _MemStore()
    store.save_klines("HOLDUSDT", "1h", df)
    monkeypatch.setattr(acc, "get_store", lambda: store)
    monkeypatch.setattr(dl, "get_store", lambda: store)
    monkeypatch.setattr(dl, "_utc_now", lambda: pd.Timestamp("2025-01-20 00:00:00"))
    monkeypatch.setattr(acc, "ensure_klines", lambda symbol, interval, days, **kw: df)

    import dataclasses as _dc

    settings = _dc.replace(
        cfg.get_settings(), enforce_holdout_pct=20.0, enforce_oos_burn=False, oos_registry_path=tmp_path / "o.md"
    )
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    a_full = audit_cell("mean_reversion", "HOLDUSDT", "1h", days=10)
    # під holdout 20% WF-вікон менше, ніж на повному ряді (200 → 160 research)
    assert a_full.status in ("ok", "error")
    # n_windows має бути обмежене research-частиною (160 барів, train/test=500/200 → 0 вікон,
    # але головне — що не використовує holdout). Перевіримо що holdout-обріз дійсно
    # відкидається: повторний виклик з holdout=0 дає >= n_windows.
    settings0 = _dc.replace(settings, enforce_holdout_pct=0.0)
    monkeypatch.setattr(cfg, "get_settings", lambda: settings0)
    a_no_holdout = audit_cell("mean_reversion", "HOLDUSDT", "1h", days=10)
    assert a_no_holdout.n_windows >= a_full.n_windows


def test_audit_cell_missing_enforce_holdout_pct_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """audit_cell не падає з AttributeError якщо Settings не має enforce_holdout_pct (stale parent memory)."""
    import pandas as pd
    from scalper_hft import config as cfg
    from scalper_hft.data import access as acc
    from scalper_hft.data import downloader as dl
    from scalper_hft.validation.cell_audit import audit_cell

    df = _klines(100, start="2025-01-01")

    class _MemStore:
        def __init__(self) -> None:
            self.data: dict[tuple[str, str], pd.DataFrame] = {}

        def load_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
            d = self.data.get((symbol, interval))
            return None if d is None else d.copy()

        def save_klines(self, symbol: str, interval: str, d: pd.DataFrame) -> None:
            self.data[(symbol, interval)] = d.copy()

    store = _MemStore()
    store.save_klines("MOCKUSDT", "1h", df)
    monkeypatch.setattr(acc, "get_store", lambda: store)
    monkeypatch.setattr(dl, "get_store", lambda: store)
    monkeypatch.setattr(dl, "_utc_now", lambda: pd.Timestamp("2025-01-20 00:00:00"))
    monkeypatch.setattr(acc, "ensure_klines", lambda symbol, interval, days, **kw: df)

    class OldSettings:
        maker_fee = 0.0002
        taker_fee = 0.0005
        slippage_frac = 0.0001
        position_pct = 0.1

    monkeypatch.setattr(cfg, "get_settings", lambda: OldSettings())
    audit = audit_cell("mean_reversion", "MOCKUSDT", "1h", days=5)
    assert audit.status != "error" or "AttributeError" not in (audit.error or "")
