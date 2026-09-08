"""Тести regime-scale overlay для pairs_arb.

Перевіряємо:
  - backward-compat: regime_scale=False → сигнали {-1,0,1}
  - regime_scale=True → дробові сигнали у [-1,1]
  - no-lookahead: мутація майбутніх барів не змінює поточний сигнал
  - entry-scale lock: розмір фіксується на вході (без resize-churn)
  - high-vol/trend leg2 → експозиція 0.5; range/normal → 1.0
  - виходи (sig==0) зберігаються
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scalper_hft.strategies.pairs_arb import PairsArb


def _cointegrated_pair(n: int = 600, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Штучно коінтегрована пара leg1 = 2*leg2 + noise + сесійний шум."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    p2 = 50.0 + np.cumsum(rng.normal(0, 0.05, n))
    spread = rng.normal(0, 0.3, n)
    # іноді більший відхил щоб породити сигнали entry_z=2
    spikes = rng.normal(0, 2.5, len(spread[::40]))
    spread[::40] += spikes
    p1 = 2.0 * p2 + 10.0 + np.cumsum(spread * 0.3) + spread
    leg1 = pd.DataFrame({"close": p1, "high": p1 + 0.5, "low": p1 - 0.5}, index=idx)
    leg2 = pd.DataFrame({"close": p2, "high": p2 + 0.3, "low": p2 - 0.3}, index=idx)
    return leg1, leg2


def _pair_df(leg1: pd.DataFrame, leg2: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"leg1": leg1["close"], "leg2": leg2["close"]}, index=leg1.index)


def test_regime_scale_off_keeps_int_signals() -> None:
    """regime_scale=False (дефолт) → сигнали лишаються {-1,0,1}."""
    leg1, leg2 = _cointegrated_pair()
    df = _pair_df(leg1, leg2)

    strat = PairsArb(entry_z=2.0, lookback=120)
    sig = strat.generate_signals(df)

    assert set(sig.dropna().unique()).issubset({-1.0, 0.0, 1.0}), "без regime_scale сигнали мають бути цілі {-1,0,1}"


def test_regime_scale_on_returns_float_in_range() -> None:
    """regime_scale=True → сигнали у [-1,1], можуть бути дробовими."""
    leg1, leg2 = _cointegrated_pair()
    df = _pair_df(leg1, leg2)

    strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=True)
    sig = strat.generate_signals(df)

    assert sig.dtype.kind == "f"
    assert sig.abs().max() <= 1.0 + 1e-9
    # має бути хоча б один ненульовий сигнал (пара коінтегрована)
    assert (sig != 0).any()


def test_regime_scale_no_lookahead() -> None:
    """Мутація барів після t не змінює сигнал на барі t."""
    leg1, leg2 = _cointegrated_pair(n=500)
    df = _pair_df(leg1, leg2)

    strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=True)
    sig = strat.generate_signals(df)

    # псуємо майбутні бари (друга половина)
    df_mut = df.copy()
    df_mut.iloc[300:] = df_mut.iloc[300:] * 3.0
    sig_mut = PairsArb(entry_z=2.0, lookback=120, regime_scale=True).generate_signals(df_mut)

    # сигнали до бару 300 мають збігатись (регим leg2 теж каузальний)
    np.testing.assert_array_equal(
        sig.iloc[:300].to_numpy(),
        sig_mut.iloc[:300].to_numpy(),
    )


def test_regime_scale_high_vol_halves_entry() -> None:
    """У high-vol leg2 новий вхід масштабується до 0.5."""
    # конструюємо leg2 з різким сплеском волатильності у другій половині
    n = 700
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(3)
    p2 = pd.Series(50.0 + np.cumsum(rng.normal(0, 0.05, n)), index=idx)
    # інжектуємо high-vol regime: великі стрибки з бару 400
    p2.iloc[400:] = p2.iloc[400:].to_numpy() + np.cumsum(rng.normal(0, 2.5, n - 400))

    p1 = pd.Series(2.0 * p2.to_numpy() + 10.0 + rng.normal(0, 0.5, n), index=idx)
    # додаємо відхил щоб породити входи
    spikes = rng.normal(0, 4.0, len(p1.iloc[::30]))
    p1.iloc[::30] = p1.iloc[::30].to_numpy() + spikes

    df = pd.DataFrame({"leg1": p1, "leg2": p2}, index=idx)

    strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=True)
    sig = strat.generate_signals(df)

    from scalper_hft.features.regimes import named_market_state

    vol = named_market_state(p2)["vol"]
    high_vol_mask = (vol == "high").to_numpy()
    # на барах high-vol, де є новий вхід (сигнал != 0), масштаб має бути 0.5
    new_entry = (sig != 0).to_numpy() & np.roll((sig == 0).to_numpy(), 1)
    new_entry[0] = False
    high_vol_entries = new_entry & high_vol_mask
    if high_vol_entries.any():
        # на хоча б одному вході у high-vol сигнал має бути 0.5 за модулем
        scaled = sig.abs().to_numpy()[high_vol_entries]
        assert np.isclose(scaled, 0.5).any() or (scaled <= 0.5 + 1e-9).all(), (
            f"у high-vol нові входи мають масштабуватись до 0.5, got {scaled}"
        )


def test_regime_scale_range_normal_keeps_full_size() -> None:
    """У range/normal leg2 вхід лишається повним (1.0)."""
    leg1, leg2 = _cointegrated_pair(seed=11)
    df = _pair_df(leg1, leg2)

    strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=True)
    sig = strat.generate_signals(df)

    from scalper_hft.features.regimes import named_market_state

    state = named_market_state(leg2["close"])
    vol = state["vol"].reindex(sig.index).fillna("normal")
    structure = state["structure"].reindex(sig.index).fillna("range")
    favorable = ((vol != "high") & (~structure.isin(["trend_up", "trend_down"]))).to_numpy()
    new_entry = (sig != 0).to_numpy() & np.roll((sig == 0).to_numpy(), 1)
    new_entry[0] = False
    fav_entries = new_entry & favorable
    if fav_entries.any():
        scaled = sig.abs().to_numpy()[fav_entries]
        assert np.isclose(scaled, 1.0).all(), f"у range/normal входи мають бути повного розміру 1.0, got {scaled}"


def test_regime_scale_exits_preserved() -> None:
    """Виходи (sig==0) не масштабуються — лишається 0."""
    leg1, leg2 = _cointegrated_pair()
    df = _pair_df(leg1, leg2)

    strat_base = PairsArb(entry_z=2.0, lookback=120)
    sig_base = strat_base.generate_signals(df)
    strat_scaled = PairsArb(entry_z=2.0, lookback=120, regime_scale=True)
    sig_scaled = strat_scaled.generate_signals(df)

    # там де base == 0 → scaled теж == 0 (виходи зберігаються)
    zero_mask = (sig_base == 0).to_numpy()
    assert (sig_scaled.to_numpy()[zero_mask] == 0).all()


def test_regime_scale_entry_size_locked_no_resize_churn() -> None:
    """Розмір фіксується на вході: під час утримання не змінюється
    навіть якщо режим leg2 змінюється (без resize-churn)."""
    leg1, leg2 = _cointegrated_pair(n=700, seed=21)
    df = _pair_df(leg1, leg2)

    strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=True)
    sig = strat.generate_signals(df)

    s = sig.to_numpy()
    # знаходимо трейд: послідовність non-zero після нуля
    for i in range(1, len(s) - 1):
        if s[i] != 0 and s[i - 1] == 0:
            # початок угоди; шукаємо її кінець
            j = i
            while j < len(s) and s[j] != 0:
                j += 1
            # усі бари угоди [i, j) мають однаковий |сигнал| (розмір зафіксований)
            trade_sizes = np.abs(s[i:j])
            if len(trade_sizes) > 1:
                assert np.allclose(trade_sizes, trade_sizes[0]), (
                    f"розмір має бути сталим під час угоди (lock at entry), got sizes={trade_sizes}"
                )
            i = j


def test_regime_scale_reproducible_across_calls() -> None:
    """Повторний виклик generate_signals на тому ж df — ідентичний результат
    (стан не витікає між викликами)."""
    leg1, leg2 = _cointegrated_pair()
    df = _pair_df(leg1, leg2)

    strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=True)
    sig1 = strat.generate_signals(df)
    sig2 = strat.generate_signals(df)

    pd.testing.assert_series_equal(sig1, sig2)


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_regime_scale_no_nan(seed: int) -> None:
    leg1, leg2 = _cointegrated_pair(seed=seed)
    df = _pair_df(leg1, leg2)
    strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=True)
    sig = strat.generate_signals(df)
    assert not sig.isna().any(), f"seed={seed}: NaN у сигналах"
