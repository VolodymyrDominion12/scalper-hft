"""`require_stream_coverage`: потік мусить перекривати період бектесту.

Контекст (аудит 2026-09-11, знахідка K2). `results/iter7_shard_*.csv` містили 20
клітинок зі `status="ok"` і Sharpe рівно 0.0: aggTrades-кеш покривав ~2 доби з
3 років, `Strategy.validate_inputs` бачив ПОВНИЙ (непорожній) фрейм і пропускав,
а walk-forward далі різав trades за часом для кожного вікна й отримував порожні
зрізи — стратегія мовчки торгувала нулями.

Тепер це помилка з конкретними числами в повідомленні.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scalper_hft.strategies.base import MissingDataError, Strategy
from scalper_hft.validation.walk_forward import (
    _stream_coverage,
    require_stream_coverage,
    run_walk_forward,
)


def _df(n: int = 3000, freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1.0}, index=idx
    )


def _stream(index: pd.DatetimeIndex, col: str = "trade_id") -> pd.DataFrame:
    if col == "trade_id":
        return pd.DataFrame(
            {"trade_id": range(len(index)), "price": 100.0, "amount": 1.0, "side": "buy"}, index=index
        )
    return pd.DataFrame({col: 0.0001, "fundingRate": 0.0001}, index=index)


class _NeedsTrades(Strategy):
    name = "needs_trades_stub"
    needs_trades = True

    def generate_signals(self, df, trades=None, funding=None):  # noqa: ARG002
        return pd.Series(1.0, index=df.index)


class _NeedsFunding(Strategy):
    name = "needs_funding_stub"
    needs_funding = True

    def generate_signals(self, df, trades=None, funding=None):  # noqa: ARG002
        return pd.Series(1.0, index=df.index)


# ── _stream_coverage ────────────────────────────────────────────────────────


def test_stream_coverage_fractions() -> None:
    df = _df(100)
    full = df.index
    assert _stream_coverage(_stream(full), df) == pytest.approx(1.0, abs=0.02)
    half = df.index[len(df) // 2 :]
    assert _stream_coverage(_stream(half), df) == pytest.approx(0.5, abs=0.02)
    # Потік ЗА межами df → нульове перекриття.
    outside = pd.date_range(df.index[-1] + pd.Timedelta(days=10), periods=10, freq="1h")
    assert _stream_coverage(_stream(outside), df) == 0.0
    assert _stream_coverage(None, df) == 0.0
    assert _stream_coverage(pd.DataFrame(), df) == 0.0


# ── require_stream_coverage ────────────────────────────────────────────────


def test_raises_when_trades_cover_tiny_fraction() -> None:
    """Реальний кейс iter7: aggTrades покривають 2 доби з 125."""
    df = _df(3000)  # 125 діб
    tail = pd.date_range(df.index[-1] - pd.Timedelta(days=2), df.index[-1], freq="1min")

    with pytest.raises(MissingDataError) as exc:
        require_stream_coverage(_NeedsTrades(), df, _stream(tail), None)

    msg = str(exc.value)
    assert "trades" in msg
    assert "1.6%" in msg or "0.3%" in msg or "1.5%" in msg, msg
    assert "50%" in msg


def test_raises_for_funding_too() -> None:
    df = _df(3000)
    tail = pd.date_range(df.index[-1] - pd.Timedelta(days=3), df.index[-1], freq="8h")

    with pytest.raises(MissingDataError) as exc:
        require_stream_coverage(_NeedsFunding(), df, None, _stream(tail, col="fundingRate"))

    assert "funding" in str(exc.value)


def test_passes_when_stream_covers_period() -> None:
    df = _df(3000)
    require_stream_coverage(_NeedsTrades(), df, _stream(df.index), None)  # не падає


def test_ignores_streams_the_strategy_does_not_need() -> None:
    """Зайвий потік (не потрібен стратегії) не має блокувати бектест."""
    df = _df(500)
    tail = pd.date_range(df.index[-1] - pd.Timedelta(hours=2), df.index[-1], freq="1min")
    require_stream_coverage(_NeedsFunding(), df, _stream(tail), _stream(df.index, col="fundingRate"))


def test_min_coverage_is_configurable() -> None:
    df = _df(1000)
    tail = df.index[-200:]
    require_stream_coverage(_NeedsTrades(), df, _stream(tail), None, min_coverage=0.1)
    with pytest.raises(MissingDataError):
        require_stream_coverage(_NeedsTrades(), df, _stream(tail), None, min_coverage=0.9)


# ── інтеграція з run_walk_forward ──────────────────────────────────────────


def test_run_walk_forward_raises_on_uncovered_stream() -> None:
    df = _df(3000)
    tail = pd.date_range(df.index[-1] - pd.Timedelta(days=1), df.index[-1], freq="1min")

    with pytest.raises(MissingDataError):
        run_walk_forward(df, _NeedsTrades(), train_bars=500, test_bars=200, trades=_stream(tail))


def test_run_walk_forward_strict_data_false_opts_out() -> None:
    """Явний opt-out лишається для дослідницьких прогонів."""
    df = _df(3000)
    tail = pd.date_range(df.index[-1] - pd.Timedelta(days=1), df.index[-1], freq="1min")

    res = run_walk_forward(
        df, _NeedsTrades(), train_bars=500, test_bars=200, trades=_stream(tail), strict_data=False
    )

    assert res.windows
