"""Тести портованих стратегій (trade-bot-main → scalper-hft).

Покриття: supertrend, stoch_rsi, smc_fvg — реєстрація, форма сигналів,
детермінізм, відсутність lookahead та smoke-бектест з комісіями.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies import REGISTRY, get_strategy

PORTED = ["supertrend", "stoch_rsi", "smc_fvg"]


def make_klines(n: int = 1500, seed: int = 7, start_price: float = 100.0) -> pd.DataFrame:
    """Детермінований OHLCV (як у test_system.py, з більшою волатильністю)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0015, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0015, n))
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


@pytest.mark.parametrize("name", PORTED)
def test_ported_registered(name: str) -> None:
    assert name in REGISTRY
    strat = get_strategy(name)
    assert strat.name == name


@pytest.mark.parametrize("name", PORTED)
@pytest.mark.parametrize("allow_short", [False, True])
def test_signals_shape_and_range(name: str, allow_short: bool) -> None:
    df = make_klines()
    strat = get_strategy(name, allow_short=allow_short)
    sig = strat.generate_signals(df)
    assert isinstance(sig, pd.Series)
    assert sig.index.equals(df.index)
    assert set(pd.unique(sig)).issubset({-1, 0, 1})
    # після warmup має бути хоча б одна позиція на такому обсязі даних
    assert (sig != 0).any()


@pytest.mark.parametrize("name", PORTED)
def test_signals_deterministic(name: str) -> None:
    df = make_klines()
    a = get_strategy(name).generate_signals(df)
    b = get_strategy(name).generate_signals(df)
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("name", PORTED)
def test_no_lookahead(name: str) -> None:
    """Сигнал на барі T залежить лише від даних до T включно.

    Якщо порахувати сигнали на обрізаному до T датафреймі — значення на
    останньому барі має збігтися з повним прогоном.
    """
    df = make_klines()
    strat = get_strategy(name)
    full = strat.generate_signals(df)
    for cut in (80, 200, 700):
        sub = strat.generate_signals(df.iloc[:cut])
        assert sub.iloc[-1] == full.iloc[cut - 1], f"{name}: lookahead на зрізі {cut}"


@pytest.mark.parametrize("name", PORTED)
def test_backtest_smoke(name: str) -> None:
    """End-to-end: бектест з повними комісіями не падає і повертає метрики."""
    df = make_klines(n=1500)
    strat = get_strategy(name)
    cost = CostModel(maker_fee=0.0002, taker_fee=0.0005, slippage_frac=0.0002)
    res = run_backtest(df, strat, cost=cost, position_pct=0.01)
    assert res.metrics.n_trades >= 0
    assert len(res.equity) == len(df)
    assert np.isfinite(res.metrics.total_return)


def test_ported_usable_in_ensemble() -> None:
    """Портовані стратегії можна одразу комбінувати у ensemble.

    Режим 'mean' усереднює сигнали суб-стратегій — допустимі часткові
    позиції (напр. 0.5), тому перевіряємо діапазон, а не дискретний набір.
    """
    df = make_klines(n=1500)
    ens = get_strategy("ensemble", strategies="supertrend,stoch_rsi", mode="mean")
    sig = ens.generate_signals(df)
    assert sig.index.equals(df.index)
    assert sig.between(-1.0, 1.0).all()
