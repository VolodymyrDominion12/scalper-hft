"""Метрики стратегії (книга, гл. 9 — "What Constitutes a Good Model?").

Окрім класичних (Sharpe, Sortino, maxDD, win rate, PF) додаємо:
    - Calmar, exposure, trades/day, avg trade return;
    - turnover — середня зміна позиції (міра витрат);
    - risk_of_ruin — імовірність розорення за edge/fraction (для оцінки ставки).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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
    first_eq = float(equity.iloc[0])
    last_eq = float(equity.iloc[-1])
    total_return = last_eq / first_eq - 1.0 if first_eq > 0 else float("nan")

    # CAGR: на вікнах ≪ 1 року (500 годинних барів ≈ 0.057 року) показник
    # степеня 1/years експлодує (×~17.5 річного множника) — це вводить в оману.
    # Для вікон < 1 року повертаємо НЕ ануалізовану дохідність (total_return),
    # щоб метрика лишалась змістовною.
    span_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = max(span_seconds / (365 * 24 * 3600), 1e-9)
    if years >= 1.0:
        if first_eq > 0 and last_eq > 0:
            cagr = (last_eq / first_eq) ** (1 / years) - 1.0
        elif first_eq > 0 and last_eq <= 0:
            # Капітал знищено (equity ≤ 0): степінь від'ємного → NaN + RuntimeWarning.
            cagr = -1.0  # повна втрата капіталу
        else:
            cagr = float("nan")  # старт ≤ 0 або нечислові значення — CAGR безглуздий
    else:
        cagr = total_return

    # Auto-detect timeframe to scale properly
    # Calculate median time delta between bars in seconds
    diffs = equity.index.to_series().diff().dropna()
    if len(diffs) and diffs.gt(pd.Timedelta(0)).all() and diffs.notna().all():
        delta_s = diffs.median().total_seconds()
    else:
        # дублікати індексу/нульові дельти: median = 0 → bars_per_year = 31.5M
        # (Sharpe/vol завищені ×5616). Fallback на хвилинний масштаб.
        delta_s = 60.0
    bars_per_year = (365 * 24 * 3600) / max(delta_s, 1.0)

    ann_vol = ret.std(ddof=0) * math.sqrt(bars_per_year) if len(ret) else 0.0
    sharpe = (ret.mean() / ret.std(ddof=0) * math.sqrt(bars_per_year)) if ret.std(ddof=0) > 0 else 0.0

    # Sortino: стандартне downside deviation = sqrt(mean(min(r,0)^2))
    # (RMS лише збиткових періодів), а не std підмножини збитків. При
    # відсутності збитків — безкінечність (ідеальна крива), а не 0.
    downside_dev = float(np.sqrt((np.minimum(ret.values, 0.0) ** 2).mean())) if len(ret) else 0.0
    if downside_dev > 0:
        sortino = ret.mean() / downside_dev * math.sqrt(bars_per_year)
    else:
        sortino = float("inf") if ret.mean() > 0 else 0.0

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
            # стандартне downside deviation (як у sortino вище), не std збитків
            down_dev = float(np.sqrt((np.minimum(hourly.values, 0.0) ** 2).mean()))
            if down_dev > 0:
                sortino_hourly = float(hourly.mean() / down_dev)
            elif hourly.mean() > 0:
                sortino_hourly = float("inf")
    except Exception:  # noqa: BLE001 — нерегулярний індекс, не критично
        pass

    # Імовірність розорення: класичне наближення для адитивного випадкового
    # блукання капіталу з дрейфом — P(ruin) = exp(-2·μ/σ²), де μ і σ — середній
    # прибуток і волатильність ПРИБУТКУ НА УГОДУ (у частках капіталу).
    # Раніше формула використовувала захардкоджену f=0.01 і μ у змішаних
    # одиницях (avg_trade_return вже масштабований position_pct), тож результат
    # був ≈константою. Тепер — лише з фактичного розподілу угод.
    if n_trades >= 5 and avg_trade_return > 0:
        sigma_trade = float(tr.std(ddof=0))
        if sigma_trade > 0:
            risk_of_ruin = math.exp(-2.0 * avg_trade_return / (sigma_trade * sigma_trade))
        else:
            risk_of_ruin = 0.0  # детермінований додатний прибуток — руїни немає
    else:
        risk_of_ruin = 1.0  # мало даних або від'ємний edge — консервативно

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
