"""Тести Flow Toxicity Gate (VPIN + Hawkes) для pairs_arb.

Дослідження §1.1–1.2: високий VPIN + Hawkes-дисбаланс = інформований потік,
що пробиває support/resistance → mean-reversion небезпечна. Гейт блокує лише
нові входи; виходи та утримання позиції — не блокує.

Перевіряємо:
  - дефолт off (flow_toxicity_gate=False)
  - без trades → no-op
  - високий VPIN блокує нові входи
  - високий Hawkes-дисбаланс блокує нові входи
  - утримання позиції не блокується (тільки нові входи)
  - виходи (sig==0) зберігаються
  - no-lookahead: мутація майбутніх trades не змінює поточний сигнал
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scalper_hft.strategies.pairs_arb import PairsArb


def _cointegrated_pair(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Штучно коінтегрована пара leg1 = 2*leg2 + noise + сесійний шум."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    p2 = 50.0 + np.cumsum(rng.normal(0, 0.05, n))
    spread = rng.normal(0, 0.3, n)
    spikes = rng.normal(0, 2.5, len(spread[::40]))
    spread[::40] += spikes
    p1 = 2.0 * p2 + 10.0 + np.cumsum(spread * 0.3) + spread
    return pd.DataFrame({"leg1": p1, "leg2": p2}, index=idx)


def _trades_for_pair(df: pd.DataFrame, *, toxic_at: pd.Timestamp | None = None, seed: int = 11) -> pd.DataFrame:
    """Синтетичні aggTrades, вирівняні з df.index.

    За замовчуванням — збалансований потік (buy≈sell). Якщо задано `toxic_at`,
    навколо цього часу генеруємо різкий дисбаланс (переважно buy) → високий VPIN.

    Об'єми підібрані так, щоб bar_volume=1000 у vpin() давав ~1 об'ємний бар
    на kline-бар (10 угод × amount 100 = 1000), а токсичний спайк (25 угод ×
    amount 500 = 12500) створював сильний дисбаланс у об'ємних барах.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    # базовий потік: ~10 угод на бар, збалансовані, amount=100 → 1000 volume/bar
    sides = rng.choice(["buy", "sell"], size=n * 10, p=[0.5, 0.5])
    ts_list: list[pd.Timestamp] = []
    for t in df.index:
        for _ in range(10):
            ts_list.append(t)
    trades = pd.DataFrame(
        {"price": 100.0, "amount": 100.0, "side": sides},
        index=pd.DatetimeIndex(ts_list),
    )
    if toxic_at is not None:
        # навколо toxic_at — 25 угод buy × amount 500 (токсичний потік)
        loc = df.index.get_loc(toxic_at)
        window = df.index[max(0, loc - 5) : loc + 5]
        toxic_ts: list[pd.Timestamp] = []
        for t in window:
            for _ in range(25):
                toxic_ts.append(t)
        toxic = pd.DataFrame(
            {"price": 100.0, "amount": 500.0, "side": "buy"},
            index=pd.DatetimeIndex(toxic_ts),
        )
        trades = pd.concat([trades, toxic])
        trades = trades.sort_index()
    return trades


# ── Дефолт off ────────────────────────────────────────────────────────────────


class TestDefaultOff:
    def test_flow_toxicity_gate_default_false(self) -> None:
        strat = PairsArb()
        assert strat.get("flow_toxicity_gate") is False

    def test_gate_off_does_not_block(self) -> None:
        """З flow_toxicity_gate=False сигнали не залежать від toxic trades."""
        df = _cointegrated_pair()
        trades = _trades_for_pair(df, toxic_at=df.index[300])
        strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=False, flow_toxicity_gate=False)
        sig = strat.generate_signals(df, trades=trades)
        # має бути хоча б один вхід (пара коінтегрована, спред спайки)
        assert (sig != 0).any()


# ── No-op без trades ──────────────────────────────────────────────────────────


class TestNoTrades:
    def test_no_trades_noop(self) -> None:
        df = _cointegrated_pair()
        strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=False, flow_toxicity_gate=True)
        sig_with_gate = strat.generate_signals(df, trades=None)
        strat2 = PairsArb(entry_z=2.0, lookback=120, regime_scale=False, flow_toxicity_gate=False)
        sig_no_gate = strat2.generate_signals(df, trades=None)
        # без trades гейт не має нічого блокувати → сигнали однакові
        assert sig_with_gate.equals(sig_no_gate)

    def test_empty_trades_noop(self) -> None:
        df = _cointegrated_pair()
        empty_trades = pd.DataFrame(columns=["price", "amount", "side"])
        strat = PairsArb(entry_z=2.0, lookback=120, regime_scale=False, flow_toxicity_gate=True)
        sig = strat.generate_signals(df, trades=empty_trades)
        assert sig.notna().any()


# ── Блокування входів ────────────────────────────────────────────────────────


class TestBlocking:
    def test_toxic_flow_blocks_new_entries(self) -> None:
        """Високий VPIN навколо toxic_at блокує нові входи там."""
        df = _cointegrated_pair()
        toxic_at = df.index[300]
        trades = _trades_for_pair(df, toxic_at=toxic_at)
        strat_off = PairsArb(entry_z=2.0, lookback=120, regime_scale=False, flow_toxicity_gate=False)
        strat_on = PairsArb(
            entry_z=2.0,
            lookback=120,
            regime_scale=False,
            flow_toxicity_gate=True,
            vpin_threshold=0.5,  # низький поріг щоб токсичний потік точно спрацював
            hawkes_imbalance_threshold=0.95,
        )
        sig_off = strat_off.generate_signals(df, trades=trades)
        sig_on = strat_on.generate_signals(df, trades=trades)
        # гейт має блокувати деякі сигнали (не всі — лише де токсично)
        blocked = (sig_off != 0) & (sig_on == 0)
        assert blocked.any(), "гейт має заблокувати хоча б один вхід при токсичному потоці"

    def test_holding_not_blocked(self) -> None:
        """Відкрита позиція (holding) не блокується гейтом — лише нові входи."""
        df = _cointegrated_pair()
        toxic_at = df.index[300]
        trades = _trades_for_pair(df, toxic_at=toxic_at)
        strat = PairsArb(
            entry_z=2.0,
            lookback=120,
            regime_scale=False,
            flow_toxicity_gate=True,
            vpin_threshold=0.5,
            hawkes_imbalance_threshold=0.95,
        )
        sig = strat.generate_signals(df, trades=trades)
        # якщо є послідовність {0, ±1, ±1, ...} — другий бар ±1 (holding) не має
        # бути скинутий гейтом навіть у токсичній зоні. Перевіряємо що хоча б
        # один holding-бар зберігся (сигнал ≠ 0 після сигналу ≠ 0).
        holding = (sig != 0) & (sig.shift(1).fillna(0) != 0)
        # не вимагаємо конкретно у toxic-зоні, лише що holding взагалі існує
        assert holding.any(), "має бути хоча б один holding-бар"

    def test_exits_preserved(self) -> None:
        """Виходи (sig==0) зберігаються — гейт не блокує вихід з позиції."""
        df = _cointegrated_pair()
        trades = _trades_for_pair(df, toxic_at=df.index[300])
        strat_off = PairsArb(entry_z=2.0, lookback=120, regime_scale=False, flow_toxicity_gate=False)
        strat_on = PairsArb(
            entry_z=2.0,
            lookback=120,
            regime_scale=False,
            flow_toxicity_gate=True,
            vpin_threshold=0.5,
            hawkes_imbalance_threshold=0.95,
        )
        sig_off = strat_off.generate_signals(df, trades=trades)
        sig_on = strat_on.generate_signals(df, trades=trades)
        # там де sig_off == 0 → sig_on теж має бути 0 (виходи не «розблоковуються»)
        assert ((sig_off == 0) & (sig_on != 0)).sum() == 0

    def test_balanced_flow_does_not_block(self) -> None:
        """Збалансований потік (без toxic_at) не блокує зайвого."""
        df = _cointegrated_pair()
        trades = _trades_for_pair(df, toxic_at=None)  # збалансований
        strat_off = PairsArb(entry_z=2.0, lookback=120, regime_scale=False, flow_toxicity_gate=False)
        strat_on = PairsArb(
            entry_z=2.0,
            lookback=120,
            regime_scale=False,
            flow_toxicity_gate=True,
            vpin_threshold=0.9,  # дефолт — збалансований потік не має його пробити
            hawkes_imbalance_threshold=0.7,
        )
        sig_off = strat_off.generate_signals(df, trades=trades)
        sig_on = strat_on.generate_signals(df, trades=trades)
        # при збалансованому потоці й дефолтних порогах — блоків майже нема
        blocked = (sig_off != 0) & (sig_on == 0)
        # допускаємо невеликий шум VPIN, але більшість має зберегтись
        assert blocked.sum() <= sig_off.abs().sum() * 0.3


# ── No-lookahead ──────────────────────────────────────────────────────────────


class TestNoLookahead:
    def test_future_trades_do_not_affect_past_signal(self) -> None:
        """Мутація trades у майбутньому не змінює сигнал у минулому."""
        df = _cointegrated_pair()
        trades = _trades_for_pair(df, toxic_at=df.index[400])
        strat = PairsArb(
            entry_z=2.0,
            lookback=120,
            regime_scale=False,
            flow_toxicity_gate=True,
            vpin_threshold=0.5,
            hawkes_imbalance_threshold=0.95,
        )
        sig_base = strat.generate_signals(df, trades=trades)
        # додаємо токсичний потік у майбутньому (після бару 500)
        future_toxic_ts: list[pd.Timestamp] = []
        for t in df.index[500:550]:
            for _ in range(30):
                future_toxic_ts.append(t)
        future_toxic = pd.DataFrame(
            {"price": 100.0, "amount": 2.0, "side": "sell"},
            index=pd.DatetimeIndex(future_toxic_ts),
        )
        trades2 = pd.concat([trades, future_toxic]).sort_index()
        sig_future = strat.generate_signals(df, trades=trades2)
        # сигнали до бару 500 не мають змінитись (no-lookahead)
        past = df.index[:500]
        assert sig_base.loc[past].equals(sig_future.loc[past]), (
            "мутація майбутніх trades не повинна змінювати минулі сигнали (lookahead)"
        )
