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

from scalper_hft.backtest.execution import CostModel
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
        rows.append({"entry_ts": entry_ts, "exit_ts": positions.index[-1], "side": int(cur_pos / abs(cur_pos)), "ret": cum_ret})
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

    close = df["close"]
    ret = close.pct_change().fillna(0.0)

    # Позиція з лагом 1: сигнал, обчислений на закритті бару t, діє з бару t+1.
    # Тому pos[t] = signals[t-1] — позиція, активна протягом бару t (без lookahead).
    pos = signals.astype(float).shift(1).fillna(0.0).clip(-1, 1)
    pos = pos * position_pct  # частка капіталу (ноціонал), знак = напрямок

    turnover = (pos - pos.shift(1)).abs().fillna(pos.abs())
    fee_rate = cost.maker_cost_per_side() if is_maker else cost.taker_cost_per_side()
    fees = turnover * fee_rate

    # Прибуток за бар t = позиція, активна в t, × дохідність бару t, мінус комісії.
    strat_ret = pos * ret - fees

    # Funding cash flow: ставка, вирівняна на бари; вплив = −позиція × ставка
    # (лонг з позитивним фандінгом платить). Ставка відома зі свого періоду.
    if funding is not None and not funding.empty:
        fr = funding["fundingRate"].reindex(df.index, method="ffill").fillna(0.0)
        strat_ret = strat_ret - pos * fr

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
