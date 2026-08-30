"""Векторизований рушій бектесту для свічкових стратегій.

Модель виконання (без lookahead):
    - сигнал обчислюється на закритті бару t;
    - позиція діє з бару t+1: strat_ret_t = pos_{t-1} × ret_t;
    - комісії та slippage сплачуються за turnover (зміну позиції).

Параметри позиції: `position_pct` — частка капіталу на угоду (ноціонал);
`max_leverage` — обмеження сумарного ноціоналу. Для ф'ючерсів позиція
відображається на ноціонал, прибуток — на різницю цін × розмір.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel, apply_breakeven_gate
from scalper_hft.backtest.metrics import BacktestMetrics, compute_metrics
from scalper_hft.strategies.base import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: BacktestMetrics
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return self.metrics.summary()


def _extract_trades(positions: pd.Series, ret: pd.Series, fees: pd.Series) -> pd.DataFrame:
    """Виділення окремих угод з позиційної серії (вхід/вихід)."""
    rows: list[dict] = []
    cur_pos = 0
    entry_ts = None
    cum_ret = 0.0
    for ts, pos in positions.items():
        if pos != cur_pos:
            if cur_pos != 0 and entry_ts is not None:
                rows.append(
                    {
                        "entry_ts": entry_ts,
                        "exit_ts": ts,
                        "side": int(cur_pos / abs(cur_pos)) if cur_pos else 0,
                        "ret": cum_ret,
                    }
                )
            if pos != 0:
                entry_ts = ts
                cum_ret = 0.0
            else:
                entry_ts = None
            cur_pos = pos
        if cur_pos != 0 and entry_ts is not None:
            bar_ret = ret.get(ts, 0.0) * cur_pos - fees.get(ts, 0.0)
            cum_ret += bar_ret
    if cur_pos != 0 and entry_ts is not None:
        rows.append(
            {"entry_ts": entry_ts, "exit_ts": positions.index[-1], "side": int(cur_pos / abs(cur_pos)), "ret": cum_ret}
        )
    return pd.DataFrame(rows, columns=["entry_ts", "exit_ts", "side", "ret"])


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10_000.0,
    cost: CostModel | None = None,
    position_pct: float = 0.01,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    is_maker: bool = False,
) -> BacktestResult:
    """Запуск бектесту стратегії на свічкових даних.

    df: DataFrame з колонками open/high/low/close/volume.
    strategy: екземпляр Strategy (generate_signals(df, trades, funding)).
    trades: aggTrades DataFrame для стратегій, що потребують потоку заявок.
    funding: DataFrame з 'fundingRate' (індекс — час ставки). Додає funding
        грошовий потік: лонг платить позитивний фандінг, шорт отримує.
    is_maker: якщо True — використання maker-комісії (лімітні ордери).
    """
    if len(df) < 30:
        raise ValueError("Замало даних для бектесту")
    cost = cost or CostModel()
    if getattr(strategy, "needs_trades", False):
        signals = strategy.generate_signals(df, trades=trades)
    elif getattr(strategy, "needs_funding", False):
        signals = strategy.generate_signals(df, funding=funding)
    else:
        signals = strategy.generate_signals(df)
    if len(signals) != len(df):
        raise ValueError("Довжина сигналів не збігається з даними")

    # Breakeven-гейт: не торгуємо, якщо очікуваний рух < round-trip витрат
    if getattr(strategy, "use_breakeven_gate", False):
        signals = apply_breakeven_gate(signals, df, cost, is_maker=is_maker)

    close = df["close"]
    ret = close.pct_change().fillna(0.0)

    # Вектор цільових позицій (з лагом 1)
    target_pos = signals.astype(float).shift(1).fillna(0.0).clip(-1, 1) * position_pct

    if is_maker:
        # Симуляція Queue Position та Adverse Selection для Maker-ордерів
        actual_pos = np.zeros(len(df))
        adverse_penalties = np.zeros(len(df))

        target_vals = target_pos.values
        close_vals = close.values
        low_vals = df["low"].values
        high_vals = df["high"].values

        # Налаштування мікроструктури
        adverse_bps = 0.0001  # 1 bps penalty for adverse selection
        prob_touch = 0.5  # 50% chance to fill if low/high equals limit

        curr_pos = 0.0
        np.random.seed(42)  # Для відтворюваності бектестів
        rands = np.random.rand(len(df))

        for i in range(1, len(df)):
            t_pos = target_vals[i]
            if t_pos != curr_pos:
                limit_px = close_vals[i - 1]
                low_px = low_vals[i]
                high_px = high_vals[i]

                filled = False
                if t_pos > curr_pos:  # Buy order
                    if low_px < limit_px:
                        filled = True
                        adverse_penalties[i] += abs(t_pos - curr_pos) * adverse_bps
                    elif low_px == limit_px and rands[i] < prob_touch:
                        filled = True
                elif t_pos < curr_pos:  # Sell order
                    if high_px > limit_px:
                        filled = True
                        adverse_penalties[i] += abs(curr_pos - t_pos) * adverse_bps
                    elif high_px == limit_px and rands[i] < prob_touch:
                        filled = True

                if filled:
                    curr_pos = t_pos

            actual_pos[i] = curr_pos

        pos = pd.Series(actual_pos, index=df.index)
        adv_penalty_series = pd.Series(adverse_penalties, index=df.index)
    else:
        # Taker - гарантований філл на close
        pos = target_pos
        adv_penalty_series = pd.Series(0.0, index=df.index)

    turnover = (pos - pos.shift(1)).abs().fillna(pos.abs())
    fee_rate = cost.maker_cost_per_side() if is_maker else cost.taker_cost_per_side()
    fees = turnover * fee_rate + adv_penalty_series

    # Прибуток за бар t = позиція, активна в t, × дохідність бару t, мінус комісії.
    strat_ret = pos * ret - fees

    # Funding cash flow: платиться ОДИН раз на період ставки (не кожен бар!).
    # Ставка, опублікована в момент fts, застосовується до позиції, активної
    # у барі, що покриває fts: funding_pnl = −pos[bar] × rate.
    # Позиція вирішена на попередньому барі — без lookahead.
    if funding is not None and not funding.empty:
        rates = funding["fundingRate"].sort_index()
        bar_idx = df.index.searchsorted(rates.index, side="right") - 1  # останній бар ≤ fts
        mask = (bar_idx >= 0) & (bar_idx < len(df))
        valid_bars = bar_idx[mask]
        valid_rates = rates.values[mask]
        funding_impact = pd.Series(0.0, index=df.index)
        if len(valid_bars):
            funding_impact.iloc[valid_bars] = -(pos.iloc[valid_bars].values * valid_rates)
        strat_ret = strat_ret + funding_impact

    equity = (1.0 + strat_ret).cumprod() * initial_capital

    trades_df = _extract_trades(pos, ret, fees)
    exposure = float((pos != 0).mean())
    metrics = compute_metrics(
        equity,
        trades=trades_df,
        exposure=exposure,
        turnover=float(turnover.sum()),
    )
    return BacktestResult(
        equity=equity,
        positions=pos,
        trades=trades_df,
        metrics=metrics,
        params={"strategy": strategy.name, "position_pct": position_pct, "is_maker": is_maker},
    )
