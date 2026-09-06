"""Регресійні тести Фази 1.5: lookahead/стан у стратегіях і рушії.

- SparseBasketArb: ваги кошика лише з минулих даних (truncation-еквівалентність);
- RegimeSupervisor: повторний виклик generate_signals дає ідентичний результат;
- engine: needs_trades + needs_funding → обидва потоки передані;
- mean_reversion: bb_period реально впливає на сигнали;
- pairs_arb: betas не "протікає" з попереднього виклику при early-return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=close.index,
    )


def test_sparse_basket_weights_no_lookahead() -> None:
    from scalper_hft.strategies.sparse_basket import SparseBasketArb

    rng = np.random.default_rng(11)
    n = 400
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    # коінтегрований тріо: спільний фактор + ідiosyncratic шум;
    # у другій половині — РІЗКА зміна структури (щоб tail-фіт ваг її "побачив")
    factor = np.cumsum(rng.normal(0, 0.01, n)) + 2.0
    a = np.exp(factor + rng.normal(0, 0.002, n)) * 100
    b = np.exp(0.8 * factor + rng.normal(0, 0.002, n)) * 50
    c = np.exp(-0.5 * factor + rng.normal(0, 0.002, n)) * 30
    # regime break: після бару 300 актив c різко змінює бету
    c[300:] = np.exp(0.9 * factor[300:] + rng.normal(0, 0.002, n - 300)) * 30
    basket = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
    df = _ohlcv(pd.Series(a, index=idx))

    strat = SparseBasketArb(lookback=60)
    sig_full = strat.generate_signals(df, basket_df=basket)
    cut = 250
    sig_part = SparseBasketArb(lookback=60).generate_signals(df.iloc[:cut], basket_df=basket.iloc[:cut])

    pd.testing.assert_series_equal(sig_full.iloc[:cut], sig_part, check_names=False)


def test_regime_supervisor_reproducible_across_calls() -> None:
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor

    rng = np.random.default_rng(42)
    n = 600
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100 + np.sin(np.arange(n) / 50) * 5 + np.cumsum(rng.normal(0, 0.3, n)), index=idx)
    df = _ohlcv(close)

    sup = RegimeSupervisor(blend_mode="contextual_hedge", hmm_fit_bars=200)
    s1 = sup.generate_signals(df)
    s2 = sup.generate_signals(df)
    pd.testing.assert_series_equal(s1, s2)


def test_engine_passes_both_trades_and_funding() -> None:
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.strategies.base import Strategy

    received: dict[str, bool] = {}

    class _Both(Strategy):
        name = "both"
        param_space: dict = {}
        needs_trades = True
        needs_funding = True

        def generate_signals(self, df, trades=None, funding=None):
            received["trades"] = trades is not None
            received["funding"] = funding is not None
            return pd.Series(0, index=df.index)

    n = 60
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100 + np.linspace(0, 1, n), index=idx)
    df = _ohlcv(close)
    trades = pd.DataFrame({"price": 100.0}, index=idx)
    funding = pd.DataFrame({"fundingRate": 0.0001}, index=idx[::8])

    run_backtest(df, _Both(), trades=trades, funding=funding)
    assert received == {"trades": True, "funding": True}


def test_mean_reversion_bb_period_changes_signals() -> None:
    from scalper_hft.strategies.mean_reversion import MeanReversionScalper

    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    # осциляція навколо 100 — mean-reversion friendly
    close = pd.Series(100 + np.sin(np.arange(n) / 8) * 3 + rng.normal(0, 0.1, n), index=idx)
    df = _ohlcv(close)

    s20 = MeanReversionScalper(bb_period=20).generate_signals(df)
    s60 = MeanReversionScalper(bb_period=60).generate_signals(df)
    assert not s20.equals(s60), "bb_period мертвий: сигнали ідентичні"


def test_pairs_arb_betas_reset_on_early_return() -> None:
    from scalper_hft.strategies.pairs_arb import PairsArb

    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    leg1 = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, n)), index=idx)
    leg2 = pd.Series(50 + np.cumsum(np.random.default_rng(2).normal(0, 0.1, n)), index=idx)
    good = pd.DataFrame({"leg1": leg1, "leg2": leg2})

    strat = PairsArb()
    strat.generate_signals(good)
    assert strat.betas is not None

    # early-return без leg-колонок → betas має скинутись, а не лишитись старим
    bad = pd.DataFrame({"close": leg1})
    strat.generate_signals(bad)
    assert strat.betas is None
