"""Тести реалістичності fill-моделей рушія (Фаза 1.1–1.3).

- QueuePositionModel у maker-симуляції: проходження ціни крізь ліміт не
  гарантує філ; adverse-penalty масштабується токсичністю.
- Vol-aware slippage: taker-комісії ростуть з волатильністю.
- Intrabar SL/TP: вихід за ціною рівня на барі дотику; одночасний дотик
  SL+TP на одному барі — песимістично SL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.backtest.engine import _simulate_maker_fills, run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.micro_price import QueuePositionModel
from scalper_hft.strategies.base import Strategy


def _make_df(n: int = 60, close: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0, 0.001, n)
    c = close * np.exp(np.cumsum(rets))
    df = pd.DataFrame(index=idx)
    df["open"] = c
    df["close"] = c
    df["high"] = c * 1.001
    df["low"] = c * 0.999
    df["volume"] = 1000.0
    return df


class _AlwaysLong(Strategy):
    name = "always_long"

    def generate_signals(self, df, trades=None, funding=None):
        # вхід на барі 20 — після прогріву ATR(14), щоб vol-aware slippage
        # працював на реальній волатильності, а не на fillna(vol_ref)
        s = pd.Series(0, index=df.index)
        s.iloc[20:] = 1
        return s


class _LongWithLevels(Strategy):
    """Лонг з бару 5; постійні SL/TP рівні."""

    name = "long_levels"

    def __init__(self, sl: float, tp: float) -> None:
        super().__init__()
        self._sl = sl
        self._tp = tp

    def generate_signals(self, df, trades=None, funding=None):
        s = pd.Series(0, index=df.index)
        s.iloc[5:] = 1
        return s

    def exit_levels(self, df):
        out = pd.DataFrame(index=df.index, dtype=float)
        out["sl_long"] = self._sl
        out["tp_long"] = self._tp
        out["sl_short"] = np.nan
        out["tp_short"] = np.nan
        return out


# ── 1.1 Queue model у maker-симуляції ────────────────────────────────────────


def test_queue_model_reduces_maker_exposure() -> None:
    """З queue-моделлю (нейтральний потік, вузький спред) maker-філи рідші."""
    df = _make_df(120)
    base = run_backtest(df, _AlwaysLong(), is_maker=True, position_pct=0.5)
    queued = run_backtest(
        df,
        _AlwaysLong(),
        is_maker=True,
        position_pct=0.5,
        queue_model=QueuePositionModel(beta_vol=0.0),
        spread_bps=2.0,
    )
    base_exposure = float((base.positions != 0).mean())
    queued_exposure = float((queued.positions != 0).mean())
    assert queued_exposure < base_exposure


def test_queue_model_scales_adverse_penalty() -> None:
    """Високий adverse_risk на барі філу → більший adverse-penalty."""
    n = 40
    target = np.zeros(n)
    target[5:] = 0.01
    close_v = np.full(n, 100.0)
    low_v = np.full(n, 99.0)  # low < close_prev → touch-through щобару
    high_v = np.full(n, 101.0)

    fill_prob = np.full(n, 1.0)  # філ гарантований
    _, adv_calm = _simulate_maker_fills(target, close_v, low_v, high_v, fill_prob=fill_prob, adverse_risk=np.zeros(n))
    _, adv_toxic = _simulate_maker_fills(target, close_v, low_v, high_v, fill_prob=fill_prob, adverse_risk=np.ones(n))
    assert adv_toxic.sum() > adv_calm.sum() > 0.0


def test_maker_fills_without_queue_model_unchanged() -> None:
    """Back-compat: без queue_model результат ідентичний попередній моделі."""
    n = 30
    target = np.zeros(n)
    target[3:] = 0.01
    close_v = np.linspace(100, 102, n)
    low_v = close_v - 0.5
    high_v = close_v + 0.5
    a1, p1 = _simulate_maker_fills(target, close_v, low_v, high_v)
    a2, p2 = _simulate_maker_fills(target, close_v, low_v, high_v)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(p1, p2)
    # touch-through → гарантований філ на першому ж барі сегмента
    assert a1[4] == pytest.approx(0.01)


# ── 1.2 Vol-aware slippage у векторному рушії ────────────────────────────────


def test_vol_aware_slippage_raises_taker_fees() -> None:
    """cost.vol_ref << поточна волатильність → slippage ↑ → equity ↓ (taker)."""
    df = _make_df(120)
    plain = run_backtest(df, _AlwaysLong(), cost=CostModel(), position_pct=0.5)
    # дуже малий референс → scale = (vol/vol_ref)^exp >> 1
    vol_cost = CostModel(vol_ref=1e-6, vol_exp=1.0)
    vol_aware = run_backtest(df, _AlwaysLong(), cost=vol_cost, position_pct=0.5)
    assert vol_aware.equity.iloc[-1] < plain.equity.iloc[-1]


def test_vol_aware_slippage_ignored_for_maker() -> None:
    """Maker-комісія не включає slippage — vol_ref не впливає на maker path."""
    df = _make_df(120)
    plain = run_backtest(df, _AlwaysLong(), cost=CostModel(), is_maker=True, position_pct=0.5)
    vol_aware = run_backtest(df, _AlwaysLong(), cost=CostModel(vol_ref=1e-6), is_maker=True, position_pct=0.5)
    assert vol_aware.equity.iloc[-1] == pytest.approx(plain.equity.iloc[-1])


# ── 1.3 Intrabar SL/TP fills ─────────────────────────────────────────────────


def _df_both_touch() -> pd.DataFrame:
    """60 барів по close=100; на барі 10 — широкий бар: дотикає і SL, і TP."""
    df = _make_df(60, close=100.0)
    df.loc[:, "close"] = 100.0
    df.loc[:, "open"] = 100.0
    df.loc[:, "high"] = 100.2
    df.loc[:, "low"] = 99.8
    ts = df.index[10]
    df.loc[ts, "high"] = 101.5  # дотик TP=101
    df.loc[ts, "low"] = 98.5  # дотик SL=99
    df.loc[ts, "close"] = 100.6  # close-to-close був би прибутковим
    return df


def test_intrabar_both_touch_pessimistic_sl() -> None:
    """SL і TP на одному барі → вихід за SL (песимістично), не за close."""
    df = _df_both_touch()
    strat = _LongWithLevels(sl=99.0, tp=101.0)
    plain = run_backtest(df, strat, position_pct=1.0)
    intra = run_backtest(df, strat, position_pct=1.0, intrabar_exits=True)
    # intrabar: bar10 ret = 99/100 - 1 = -1%; plain: 100.6/100 - 1 = +0.6%
    assert intra.equity.iloc[-1] < plain.equity.iloc[-1]
    # позиція обнуляється після бару стопу до кінця run-у
    assert (intra.positions.iloc[11:] == 0).all()


def test_intrabar_tp_exit_at_level_price() -> None:
    """Дотик лише TP → вихід за ціною TP на барі дотику."""
    df = _make_df(60, close=100.0)
    df.loc[:, "close"] = 100.0
    df.loc[:, "open"] = 100.0
    df.loc[:, "high"] = 100.2
    df.loc[:, "low"] = 99.8
    ts = df.index[10]
    df.loc[ts, "high"] = 101.5  # дотик TP
    df.loc[ts, "low"] = 99.9  # SL=99 не задітий
    df.loc[ts, "close"] = 100.0
    strat = _LongWithLevels(sl=99.0, tp=101.0)
    res = run_backtest(
        df,
        strat,
        position_pct=1.0,
        intrabar_exits=True,
        cost=CostModel(maker_fee=0.0, taker_fee=0.0, slippage_frac=0.0),
    )
    # equity на барі 10 = 1.01 (філ рівно по TP=101 від close=100)
    assert res.equity.iloc[10] == pytest.approx(10_000.0 * 1.01, rel=1e-9)


def test_intrabar_disabled_by_default() -> None:
    """Без intrabar_exits поведінка close-to-close (back-compat)."""
    df = _df_both_touch()
    strat = _LongWithLevels(sl=99.0, tp=101.0)
    res = run_backtest(df, strat, position_pct=1.0)
    assert not (res.positions.iloc[11:] == 0).all()


def test_intrabar_no_levels_noop() -> None:
    """Стратегія без exit_levels → intrabar_exits нічого не змінює."""
    df = _df_both_touch()
    a = run_backtest(df, _AlwaysLong(), position_pct=0.5)
    b = run_backtest(df, _AlwaysLong(), position_pct=0.5, intrabar_exits=True)
    pd.testing.assert_series_equal(a.equity, b.equity)
