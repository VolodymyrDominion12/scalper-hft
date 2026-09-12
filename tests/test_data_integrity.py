"""Регресійні тести цілісності даних і дегенеративних клітинок.

Контекст: дослідницький кеш був скачаний з Binance **testnet** (синтетична
історія: ціни розходились з ринком на 1–17%, рухи +27% за 1m, «плити»
O=H=L=C з нульовим обсягом, funding обрізаний до ~13 міс). Формальна
валідація цього не бачила, sweep рахував фальшивий edge, а Bailey–LdP
haircut упевнено обрав `ml_strategy` переможцем.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scalper_hft.config import get_settings
from scalper_hft.data.downloader import funding_coverage_ratio
from scalper_hft.data.validate import (
    BarQualityReport,
    bars_are_critical,
    validate_bars,
)
from scalper_hft.research.sweep_store import SweepRow
from scalper_hft.validation.sweep import flag_degenerate_row

# ── Допоміжні генератори ─────────────────────────────────────────────────────


def _ohlcv(closes: np.ndarray, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range(end="2024-06-01", periods=len(closes), freq=freq)
    c = pd.Series(closes, index=idx)
    return pd.DataFrame(
        {"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": 1.0},
        index=idx,
    )


def _clean_closes(n: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))


# ── validate_bars: неринкові рухи ────────────────────────────────────────────


def test_validate_bars_accepts_clean_series() -> None:
    rep = validate_bars(_ohlcv(_clean_closes()), interval="1m")
    assert rep.ok
    assert rep.n_spikes == 0
    assert not bars_are_critical(rep)


def test_validate_bars_flags_synthetic_spikes() -> None:
    """Серія з 1% барів-стрибків (як testnet BNBUSDT) — критично."""
    closes = _clean_closes()
    rng = np.random.default_rng(1)
    pos = rng.choice(len(closes), 50, replace=False)
    closes[pos] *= 1.3
    rep = validate_bars(_ohlcv(closes), interval="1m")
    assert not rep.ok
    assert rep.n_spikes > 0
    assert rep.spike_rate > 0.001
    assert bars_are_critical(rep)


def test_validate_bars_flags_flat_plateau() -> None:
    """«Плита» O=H=L=C на сотні барів (нульовий обсяг на testnet) — критично."""
    closes = _clean_closes()
    closes[1000:1400] = closes[999]
    rep = validate_bars(_ohlcv(closes), interval="1m")
    assert not rep.ok
    assert rep.max_flat_run >= 400
    assert bars_are_critical(rep)


def _ohlcv_with_volume(closes: np.ndarray, volumes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range(end="2024-06-01", periods=len(closes), freq="1min")
    c = pd.Series(closes, index=idx)
    return pd.DataFrame(
        {"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": volumes},
        index=idx,
    )


def test_validate_bars_tolerates_zero_volume_halt() -> None:
    """Біржовий halt: ~1 година без угод (LINKUSDT 2021-03-02, підтверджено
    live-перекачкою) — close «заморожений», але це реальні дані біржі."""
    closes = _clean_closes()
    volumes = np.ones(len(closes))
    closes[1000:1060] = closes[999]  # 60 барів підряд
    volumes[1000:1060] = 0.0
    rep = validate_bars(_ohlcv_with_volume(closes, volumes), interval="1m")
    assert rep.ok
    assert rep.max_flat_run >= 60
    assert not bars_are_critical(rep)


def test_validate_bars_flags_traded_flat_run() -> None:
    """«Плита» серед ТОРГОВАНИХ барів (обсяг є, ціна не рухається) — неринково."""
    closes = _clean_closes()
    closes[1000:1060] = closes[999]  # 60 барів з обсягом 1.0
    rep = validate_bars(_ohlcv(closes), interval="1m")
    assert not rep.ok
    assert any("торгованих" in issue for issue in rep.issues)


def test_validate_bars_zero_volume_plateau_still_critical() -> None:
    """Багатогодинна «плита» (≥240 барів) — критично навіть з нульовим обсягом."""
    closes = _clean_closes()
    volumes = np.ones(len(closes))
    closes[1000:1400] = closes[999]
    volumes[1000:1400] = 0.0
    rep = validate_bars(_ohlcv_with_volume(closes, volumes), interval="1m")
    assert not rep.ok
    assert bars_are_critical(rep)


def test_validate_bars_tolerates_isolated_real_tails() -> None:
    """Поодинокі справжні хвости (0.002%, як NEARUSDT на live) не блокують."""
    closes = _clean_closes(200_000, seed=7)
    closes[50_000] *= 1.12
    closes[150_000] *= 0.90
    rep = validate_bars(_ohlcv(closes), interval="1m")
    assert rep.spike_rate < 0.001
    assert not bars_are_critical(rep)


def test_bars_are_critical_empty_is_critical() -> None:
    assert bars_are_critical(BarQualityReport(0, 0, 0, 0, 0, True, False, ["порожній датасет"]))


# ── funding coverage ─────────────────────────────────────────────────────────


def _funding_index(start: str, n: int, freq: str = "8h") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


def test_funding_coverage_full_window() -> None:
    now = pd.Timestamp("2026-09-10 00:00")
    idx = _funding_index("2023-09-11 00:00", int(1095 * 3) + 1)
    df = pd.DataFrame({"fundingRate": np.zeros(len(idx))}, index=idx)
    ratio = funding_coverage_ratio(df, 1095, now=now)
    assert ratio == pytest.approx(1.0, abs=0.02)


def test_funding_coverage_detects_truncated_history() -> None:
    """Testnet: свіжий кінець, але початок лише 415 днів тому → ~38%."""
    now = pd.Timestamp("2026-09-10 00:00")
    idx = _funding_index("2025-07-21 16:00", int(415 * 3) + 1)
    df = pd.DataFrame({"fundingRate": np.zeros(len(idx))}, index=idx)
    ratio = funding_coverage_ratio(df, 1095, now=now)
    assert ratio < 0.5


def test_funding_coverage_historical_cache_anchors_to_its_end() -> None:
    """Кеш із фіксованими (минулими) датами не має давати хибний FAIL."""
    idx = _funding_index("2024-01-01 00:00", 31)  # 10 діб по 8h
    df = pd.DataFrame({"fundingRate": np.zeros(len(idx))}, index=idx)
    ratio = funding_coverage_ratio(df, 10, now=pd.Timestamp("2026-09-10 00:00"))
    assert ratio >= 0.9


def test_funding_coverage_zero_days_is_neutral() -> None:
    assert funding_coverage_ratio(pd.DataFrame(), 0) == 1.0
    assert funding_coverage_ratio(None, 0) == 1.0


# ── Дегенеративні клітинки sweep ─────────────────────────────────────────────


def _row(**kwargs: object) -> SweepRow:
    base = {
        "strategy": "s",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "days": 1095,
        "mode": "backtest",
        "n_bars": 315_361,
        "n_trades": 5000,
        "total_return": 0.1,
        "sharpe": 1.2,
        "profit_factor": 1.4,
    }
    return SweepRow(**{**base, **kwargs})  # type: ignore[arg-type]


def test_flag_degenerate_zero_trades() -> None:
    row = flag_degenerate_row(_row(n_trades=0))
    assert row.status == "degenerate"
    assert "0 угод" in row.error


def test_flag_degenerate_capital_destroyed() -> None:
    row = flag_degenerate_row(_row(total_return=-1.0, sharpe=-3.0))
    assert row.status == "degenerate"
    assert "капітал знищено" in row.error


def test_flag_degenerate_implausible_sharpe() -> None:
    """ml_strategy BNBUSDT 5m давав Sharpe 56.9 (артефакт testnet-історії)."""
    row = flag_degenerate_row(_row(sharpe=56.85))
    assert row.status == "degenerate"
    assert "неправдоподібно" in row.error


def test_flag_degenerate_infinite_profit_factor() -> None:
    row = flag_degenerate_row(_row(profit_factor=float("inf")))
    assert row.status == "degenerate"


def test_flag_degenerate_keeps_healthy_cell() -> None:
    row = flag_degenerate_row(_row())
    assert row.status == "ok"
    assert row.error == ""


def test_flag_degenerate_skips_smoke_scale_runs() -> None:
    """На smoke-тестах (дні=2) 0 угод — шум вибірки, не дефект."""
    row = flag_degenerate_row(_row(days=2, n_trades=0, sharpe=40.0))
    assert row.status == "ok"


def test_flag_degenerate_walkforward_ignores_backtest_only_metrics() -> None:
    """WF-режим: total_return/profit_factor NaN за дизайном — не дефект."""
    row = flag_degenerate_row(
        _row(mode="walkforward", total_return=float("nan"), profit_factor=float("nan"), n_trades=500)
    )
    assert row.status == "ok"


def test_flag_degenerate_does_not_touch_errors() -> None:
    row = flag_degenerate_row(_row(status="error", error="boom", n_trades=0))
    assert row.status == "error"
    assert row.error == "boom"


# ── market_maker: капітал не може стати від'ємним ────────────────────────────


def test_event_engine_equity_never_negative() -> None:
    """Раніше market_maker давав total_return = −3415% (equity < 0)."""
    from scalper_hft.backtest.event_engine import run_event_backtest
    from scalper_hft.strategies import get_strategy

    idx = pd.date_range(end="2024-06-01", periods=4000, freq="5min")
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, 4000))), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 10.0,
        },
        index=idx,
    )
    strat = get_strategy("market_maker")
    res = run_event_backtest(df, strat, initial_capital=10_000.0, quote_size_pct=0.5, inventory_cap=20.0)
    assert (res.equity >= 0).all(), "equity пішла в мінус — немає стопу на ліквідацію"
    assert res.metrics.total_return >= -1.0 - 1e-9


# ── Вердикти: журнал не засмічується дублями ─────────────────────────────────


def test_record_verdict_dedupes_identical_rows(tmp_path) -> None:
    from scalper_hft.validation.verdict_store import record_verdict

    p = tmp_path / "v.jsonl"
    for _ in range(5):
        record_verdict("mean_reversion", "BTCUSDT", "1h", "FAIL", "n_trades_oos=nan<30", path=p)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1

    # Інший вердикт для тієї ж комірки — пишеться.
    record_verdict("mean_reversion", "BTCUSDT", "1h", "FAIL", "інша причина", path=p)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2

    # dedupe=False — примусовий запис.
    record_verdict("mean_reversion", "BTCUSDT", "1h", "FAIL", "інша причина", path=p, dedupe=False)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3


def test_cell_verdict_reports_audit_error() -> None:
    from scalper_hft.validation.cell_audit import cell_verdict

    label, reasons = cell_verdict({"status": "error", "error": "немає даних", "interval": "5m"})
    assert label == "FAIL"
    assert "немає даних" in reasons


# ── data-audit: звірка з live ────────────────────────────────────────────────


def test_data_audit_detects_divergence_from_live(tmp_path, monkeypatch) -> None:
    """Якщо кеш не з тієї біржі — audit_symbol це бачить."""
    from scalper_hft.data import audit as audit_mod

    closes = _clean_closes(2000, seed=11)
    idx = pd.date_range(end="2024-06-01", periods=2000, freq="1min")
    cached = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": 1.0,
        },
        index=idx,
    )

    class _Store:
        def load_klines(self, symbol: str, interval: str) -> pd.DataFrame:
            return cached

        def load_funding(self, symbol: str) -> pd.DataFrame | None:
            return None

    class _Client:
        """LIVE віддає іншу ціну (кеш — testnet)."""

        def fetch_klines(self, symbol: str, timeframe: str, since_ms: int, limit: int = 1000):
            ts = pd.Timestamp(since_ms, unit="ms").ceil("1min")
            if ts not in cached.index:
                return []
            return [[int(ts.value // 1_000_000), 0.0, 0.0, 0.0, float(cached.loc[ts, "close"]) * 1.2, 0.0]]

    res = audit_mod.audit_symbol(
        "BTCUSDT",
        interval="1m",
        days=2,
        live_samples=4,
        check_funding=False,
        store=_Store(),
        client=_Client(),
    )
    assert res.live_checked > 0
    assert res.live_mismatched == res.live_checked
    assert res.max_live_divergence > audit_mod.LIVE_DIVERGENCE_MAX
    assert not res.ok


def test_data_audit_symbol_uses_data_exchange_default() -> None:
    settings = get_settings()
    assert not settings.is_data_exchange_testnet
    assert "testnet" not in settings.data_exchange.lower()


def test_settings_data_exchange_separate_from_trading() -> None:
    s = replace(get_settings(), exchange="binance-testnet", data_exchange="binanceusdm")
    assert "testnet" in s.exchange.lower()
    assert not s.is_data_exchange_testnet
