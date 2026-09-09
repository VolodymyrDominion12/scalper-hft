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

    strat = PairsArb(regime_scale=False)
    strat.generate_signals(good)
    assert strat.betas is not None

    # early-return без leg-колонок → betas має скинутись, а не лишитись старим
    bad = pd.DataFrame({"close": leg1})
    strat.generate_signals(bad)
    assert strat.betas is None


def test_hmm_regime_features_default_is_causal() -> None:
    """Дефолт hmm_regime_features має бути causal=True (без lookahead)."""
    import inspect

    from scalper_hft.features.hmm_regime import hmm_regime_features

    sig = inspect.signature(hmm_regime_features)
    assert sig.parameters["causal"].default is True, "causal має бути True за замовчуванням"


def test_hmm_regime_features_causal_no_lookahead() -> None:
    """causal=True: дописування майбутніх барів не змінює вже видані значення.

    Головна ознака відсутності lookahead: прогноз у момент t залежить лише від
    даних ≤ t. Тому префікс ряду, обчислений на повному ряду, має збігатися з
    рядом, обчисленим лише на префіксі (на спільному відрізку).
    """
    from scalper_hft.features.hmm_regime import hmm_regime_features

    rng = np.random.default_rng(5)
    n = 600
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    # два режими: низьковол + високовол блоки
    vol = np.where(np.arange(n) % 200 < 100, 0.002, 0.02)
    rets = rng.normal(0, vol, n)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)

    out_full = hmm_regime_features(close, n_states=3, causal=True, fit_window=200, seed=7)
    cut = 350
    out_part = hmm_regime_features(close.iloc[:cut], n_states=3, causal=True, fit_window=200, seed=7)

    pcols = [c for c in out_full.columns if c.startswith("hmm_p")]
    # на спільному відрізку після прогріву ймовірності мають збігатись
    common = out_full.index.intersection(out_part.index)
    # лишаємо лише барі після завершення fit_window у частковому ряду
    warm = out_part.index[out_part.index.get_indexer(out_part.index) >= 200]
    common = common.intersection(warm)
    for c in pcols:
        a = out_full.loc[common, c].to_numpy()
        b = out_part.loc[common, c].to_numpy()
        assert np.allclose(a, b, atol=1e-8), f"lookahead у {c}: майбутні барі змінили прогноз"


def test_live_hmm_gate_uses_regime_detector_parity() -> None:
    """LiveTrader.hmm_blocked делегує до RegimeDetector (та сама політика, що backtest).

    Демонструє уніфікацію: backtest (RegimeSupervisor) і live (LiveTrader HMM-гейт)
    використовують єдиний RegimeDetector → єдина політика fit (перші 2000 барів) +
    filtered_proba (forward-only). Результати calm-prob у backtest і live збігаються.
    """
    from scalper_hft.features.regime_detector import RegimeDetector
    from scalper_hft.live.account import PaperAccount
    from scalper_hft.live.trader import LiveTrader

    rng = np.random.default_rng(9)
    n = 600
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    vol = np.where(np.arange(n) % 200 < 100, 0.002, 0.02)
    rets = rng.normal(0, vol, n)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    df = _ohlcv(close)

    class _Flat:
        name = "flat"
        param_space: dict = {}
        needs_trades = False
        needs_funding = False

        def generate_signals(self, d, trades=None, funding=None):
            return pd.Series(0, index=d.index)

    # backtest-шлях: RegimeDetector напряму
    det = RegimeDetector(n_hmm_states=3, hmm_fit_bars=2000, hmm_seed=42)
    state_df = det.detect(close)
    calm = det.hmm_calm_state
    assert calm is not None
    backtest_calm_prob = float(state_df[f"hmm_p{calm}"].iloc[-1])

    # live-шлях: LiveTrader.hmm_blocked (делегує до RegimeDetector)
    trader = LiveTrader(_Flat(), "BTCUSDT", "1h", account=PaperAccount(10_000.0), hmm_block=True, hmm_threshold=0.5)
    trader.hmm_blocked(df)  # ініціалізує внутрішній RegimeDetector
    live_det = trader._regime_detector
    assert isinstance(live_det, RegimeDetector), "live має делегувати до RegimeDetector"
    live_calm_prob = float(live_det.detect(close)[f"hmm_p{calm}"].iloc[-1])

    assert np.isclose(backtest_calm_prob, live_calm_prob, atol=1e-10), (
        "backtest і live HMM дають різний calm-prob — розходження політики fit"
    )


def test_ml_features_no_lookahead_truncation_equivalence() -> None:
    """ML-фічі (_build_features) каузальні: префікс ряду збігається з повним рядом.

    Головна ознака відсутності lookahead у фічах: значення фічі на барі t
    залежить лише від даних ≤ t. Тому `_build_features(full).iloc[:n]` має
    дорівнювати `_build_features(full.iloc[:n])` на спільному відрізку.

    Тестує базовий шлях (add_standard_features + frac_diff + cvd-константа при
    trades=None) — найчутливіший до lookahead. HMM/GARCH каузальність перевіряється
    окремо (test_hmm_regime_features_causal_no_lookahead + volatility-тести).
    """
    from scalper_hft.ml.features import _build_features

    rng = np.random.default_rng(13)
    n = 500
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, n)), index=idx)
    df = _ohlcv(close)

    feats_full = _build_features(
        df, trades=None, add_frac=True, frac_d=0.4, add_micro=False, add_hmm=False, add_garch=False
    )
    cut = 300
    feats_part = _build_features(
        df.iloc[:cut], trades=None, add_frac=True, frac_d=0.4, add_micro=False, add_hmm=False, add_garch=False
    )

    common = feats_full.index.intersection(feats_part.index)
    # порівнюємо всі спільні колонки
    cols = [c for c in feats_full.columns if c in feats_part.columns]
    for c in cols:
        a = feats_full.loc[common, c].to_numpy()
        b = feats_part.loc[common, c].to_numpy()
        assert np.allclose(a, b, equal_nan=True, atol=1e-10), (
            f"lookahead у ML-фічі {c!r}: значення на префіксі відрізняється від повного ряду"
        )
