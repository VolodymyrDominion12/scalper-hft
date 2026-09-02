"""Метрики стратегії (книга, гл. 9 — "What Constitutes a Good Model?").

Окрім класичних (Sharpe, Sortino, maxDD, win rate, PF) додаємо:
    - Calmar, exposure, trades/day, avg trade return;
    - turnover — середня зміна позиції (міра витрат);
    - risk_of_ruin — імовірність розорення за edge/fraction (для оцінки ставки).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

_TRADING_DAYS = 365


@dataclass
class BacktestMetrics:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade_return: float
    exposure: float
    trades_per_day: float
    turnover: float
    risk_of_ruin: float
    sharpe_hourly: float = 0.0  # per-period Sharpe (годинний), без ануалізації
    sortino_hourly: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Загальна дохідність:     {self.total_return:>10.2%}",
            f"CAGR:                    {self.cagr:>10.2%}",
            f"Річна волатильність:     {self.ann_vol:>10.2%}",
            f"Sharpe (річний):         {self.sharpe:>10.3f}",
            f"Sharpe (годинний):       {self.sharpe_hourly:>10.3f}  ← без ануалізації",
            f"Sortino:                 {self.sortino:>10.3f}",
            f"Calmar:                  {self.calmar:>10.3f}",
            f"Макс. просідання:        {self.max_drawdown:>10.2%}",
            f"Win rate:                {self.win_rate:>10.2%}",
            f"Profit factor:           {self.profit_factor:>10.3f}",
            f"Угоди:                   {self.n_trades:>10d}",
            f"Середній прибуток/угода: {self.avg_trade_return:>10.4%}",
            f"Експозиція:              {self.exposure:>10.2%}",
            f"Угод на день:            {self.trades_per_day:>10.2f}",
            f"Turnover (зміна позиції):{self.turnover:>10.4f}",
            f"P(розорення):            {self.risk_of_ruin:>10.4f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    exposure: float = 0.0,
    turnover: float = 0.0,
    trade_return: pd.Series | None = None,
) -> BacktestMetrics:
    """Обчислення метрик за кривою капіталу (equity) і списком угод.

    equity: Series з індексом datetime, значення — капітал.
    trades: DataFrame з колонками ['entry_ts','exit_ts','ret'] (опційно).
    trade_return: Series прибутковостей окремих угод (альтернатива trades).
    """
    if equity is None or len(equity) < 2:
        raise ValueError("equity має містити щонайменше 2 точки")

    ret = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    years = max((equity.index[-1] - equity.index[0]).total_seconds() / (365 * 24 * 3600), 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0

    # Auto-detect timeframe to scale properly
    # Calculate median time delta between bars in seconds
    delta_s = equity.index.to_series().diff().median().total_seconds()
    bars_per_year = (365 * 24 * 3600) / max(delta_s, 1.0) if pd.notna(delta_s) else _TRADING_DAYS * 24 * 60

    ann_vol = ret.std(ddof=0) * math.sqrt(bars_per_year) if len(ret) else 0.0
    sharpe = (ret.mean() / ret.std(ddof=0) * math.sqrt(bars_per_year)) if ret.std(ddof=0) > 0 else 0.0

    downside = ret[ret < 0]
    sortino = (
        (ret.mean() / downside.std(ddof=0) * math.sqrt(bars_per_year))
        if len(downside) > 1 and downside.std(ddof=0) > 0
        else 0.0
    )

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_drawdown = dd.min() if len(dd) else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # угоди
    if trades is not None and not trades.empty and "ret" in trades.columns:
        tr = trades["ret"]
        n_trades = len(tr)
    elif trade_return is not None and len(trade_return):
        tr = trade_return
        n_trades = len(tr)
    else:
        tr = pd.Series(dtype=float)
        n_trades = 0

    if n_trades:
        wins = tr[tr > 0]
        losses = tr[tr < 0]
        win_rate = len(wins) / n_trades
        gross_win = wins.sum()
        gross_loss = -losses.sum()
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        avg_trade_return = tr.mean()
    else:
        win_rate = profit_factor = avg_trade_return = 0.0

    span_days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400.0, 1.0)
    trades_per_day = n_trades / span_days

    # per-period метрики (годинні, БЕЗ ануалізації) — чесне порівняння між таймфреймами
    sharpe_hourly = 0.0
    sortino_hourly = 0.0
    try:
        hourly = equity.resample("1h").last().pct_change().dropna()
        if len(hourly) >= 2 and hourly.std(ddof=0) > 0:
            sharpe_hourly = float(hourly.mean() / hourly.std(ddof=0))
            down = hourly[hourly < 0]
            if len(down) >= 2 and down.std(ddof=0) > 0:
                sortino_hourly = float(hourly.mean() / down.std(ddof=0))
    except Exception:  # noqa: BLE001 — нерегулярний індекс, не критично
        pass

    # імовірність розорення за спрощеною формулою Келлі P(ruin) = exp(-2 * edge * N)
    # де N = capital / stake. Для фіксованого f = 0.01, N = 100
    edge = avg_trade_return
    f = 0.01  # частка капіталу на угоду (конфігурується окремо)
    if edge > 0:
        # Експоненційне наближення (більш реалістичне для трейдингу)
        risk_of_ruin = math.exp(-2.0 * edge * (1.0 / f))
    else:
        risk_of_ruin = 1.0

    return BacktestMetrics(
        total_return=total_return,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=n_trades,
        avg_trade_return=avg_trade_return,
        exposure=exposure,
        trades_per_day=trades_per_day,
        turnover=turnover,
        risk_of_ruin=risk_of_ruin,
        sharpe_hourly=sharpe_hourly,
        sortino_hourly=sortino_hourly,
    )
