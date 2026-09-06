"""Кеш фіч indicators + TTL-кеш/HMM-refit у live-кроці."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _df(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(3).normal(0, 0.1, n)), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=idx,
    )


def test_add_standard_features_cache_same_result() -> None:
    from scalper_hft.features.indicators import add_standard_features, clear_features_cache

    clear_features_cache()
    df = _df()
    a = add_standard_features(df)
    b = add_standard_features(df)  # з кешу
    pd.testing.assert_frame_equal(a, b)


def test_add_standard_features_cache_returns_copy() -> None:
    """Мутація результату не забруднює кеш."""
    from scalper_hft.features.indicators import add_standard_features, clear_features_cache

    clear_features_cache()
    df = _df()
    a = add_standard_features(df)
    a["bb_mid"] = -1.0  # мутація копії caller'а
    b = add_standard_features(df)
    assert (b["bb_mid"] != -1.0).all()


def test_add_standard_features_cache_disabled() -> None:
    from scalper_hft.features.indicators import add_standard_features, clear_features_cache

    clear_features_cache()
    df = _df()
    a = add_standard_features(df, use_cache=False)
    b = add_standard_features(df)
    pd.testing.assert_frame_equal(a, b)


def test_live_trader_aux_data_ttl_cache() -> None:
    """funding/aggTrades у live-кроці кешуються: повторний виклик без завантаження."""
    from types import SimpleNamespace

    from scalper_hft.live.account import PaperAccount
    from scalper_hft.live.trader import LiveTrader

    calls = {"n": 0}
    idx = pd.date_range("2025-01-01", periods=80, freq="1min")
    funding_df = pd.DataFrame({"fundingRate": 0.0001}, index=idx[::8])

    class _FundingStrat:
        name = "f"
        param_space: dict = {}
        needs_trades = False
        needs_funding = True

        def generate_signals(self, df, trades=None, funding=None):
            return pd.Series(0, index=df.index)

    import scalper_hft.data.downloader as dl

    orig = dl.download_funding

    def _spy(symbol, days=30):
        calls["n"] += 1
        return funding_df

    try:
        dl.download_funding = _spy
        # trader.py імпортує downloader всередині методу — патчимо модуль
        trader = LiveTrader(_FundingStrat(), "BTCUSDT", "1m", account=PaperAccount(10_000.0))
        trader.settings = SimpleNamespace(dry_run=True)
        df = _df(80)
        trader.compute_signal(df)
        trader.compute_signal(df)
        assert calls["n"] == 1, f"очікував 1 виклик завантаження, отримав {calls['n']}"
    finally:
        dl.download_funding = orig


def test_hmm_refit_cached(monkeypatch) -> None:
    """HMM fit викликається не частіше ніж раз на _HMM_REFIT_BARS барів."""
    from types import SimpleNamespace

    from scalper_hft.live.account import PaperAccount
    from scalper_hft.live.trader import LiveTrader

    fits = {"n": 0}

    class _FakeHMM:
        def __init__(self, n_states=3, seed=42):
            self.covars_ = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])

        def fit(self, X):
            fits["n"] += 1
            return self

        def filtered_proba(self, X):
            return np.tile([0.8, 0.1, 0.1], (len(X), 1))

    import scalper_hft.features.hmm_regime as hmm_mod

    monkeypatch.setattr(hmm_mod, "GaussianHMM", _FakeHMM)

    class _Flat:
        name = "flat"
        param_space: dict = {}
        needs_trades = False
        needs_funding = False

        def generate_signals(self, df, trades=None, funding=None):
            return pd.Series(0, index=df.index)

    trader = LiveTrader(_Flat(), "BTCUSDT", "1m", account=PaperAccount(10_000.0), hmm_block=True)
    trader.settings = SimpleNamespace(dry_run=True)

    df = _df(300)
    trader.hmm_blocked(df)
    trader.hmm_blocked(df)  # той самий обсяг даних — без refit
    assert fits["n"] == 1
    df2 = _df(300 + LiveTrader._HMM_REFIT_BARS + 10)
    trader.hmm_blocked(df2)  # дані "виросли" — refit
    assert fits["n"] == 2
