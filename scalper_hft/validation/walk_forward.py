"""Walk-Forward Analysis (книга, гл. 9; дослідження: WFA — стандарт індустрії).

Ковзні вікна train (IS) → test (OOS):
    [====TRAIN_1====][OOS_1]
          [====TRAIN_2====][OOS_2]
                [====TRAIN_3====][OOS_3]

Головна метрика: середній OOS Sharpe та частка вікон з OOS Sharpe > 0.
Якщо IS Sharpe високий, а OOS — біля нуля/негативний → стратегія
перенавчена (книга, гл. 11 — "Quants Are Guilty of Data Mining").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scalper_hft.backtest.engine import run_backtest
from scalper_hft.backtest.execution import CostModel
from scalper_hft.strategies.base import Strategy

if TYPE_CHECKING:
    from scalper_hft.overlay.policy import CellPolicy


@dataclass
class WalkForwardWindow:
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    is_sharpe: float
    oos_sharpe: float
    oos_return: float
    n_trades: int


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    avg_is_sharpe: float
    avg_oos_sharpe: float
    positive_windows_frac: float
    degradation: float
    details: dict = field(default_factory=dict)
    # Конкатеновані OOS бар-дохідності (лише якщо collect_oos_returns=True) —
    # для чесного DSR/PBO на OOS, без IS-забруднення повної вибірки.
    oos_returns: pd.Series | None = None

    def summary(self) -> str:
        lines = [
            "Walk-Forward Analysis",
            f"  вікон:                {len(self.windows)}",
            f"  avg IS Sharpe:        {self.avg_is_sharpe:.3f}",
            f"  avg OOS Sharpe:       {self.avg_oos_sharpe:.3f}",
            f"  частка вікон OOS>0:   {self.positive_windows_frac:.0%}",
            f"  деградація IS→OOS:    {self.degradation:.1%}",
        ]
        for w in self.windows:
            lines.append(
                f"  [{w.window_idx}] IS {w.train_start}:{w.train_end} → OOS {w.test_start}:{w.test_end} "
                f"| IS SR {w.is_sharpe:+.2f} | OOS SR {w.oos_sharpe:+.2f} | ret {w.oos_return:+.2%} | {w.n_trades} угод"
            )
        return "\n".join(lines)


def _sharpe_from_equity(equity: pd.Series) -> float:
    """Per-window Sharpe у t-stat-подібній шкалі: mean/std × √n_барів вікна.

    УВАГА: шкала відрізняється від BacktestMetrics.sharpe (ануалізований,
    metrics.py) — пороги на кшталт OOS_SHARPE_MIN у cell_audit стосуються
    САМЕ цієї per-window шкали і не порівнянні з bt_sharpe напряму.
    Використовується для порівняння IS та OOS вікон однакового розміру.
    """
    ret = equity.pct_change().dropna()
    if len(ret) < 2 or ret.std(ddof=0) == 0:
        return 0.0
    return float(ret.mean() / ret.std(ddof=0) * np.sqrt(len(ret)))


def _slice_by_time(funding: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """Зріз funding за часовим діапазоном [t0, t1]."""
    mask = (funding.index >= t0) & (funding.index <= t1)
    return funding[mask]


def run_walk_forward(
    df: pd.DataFrame,
    strategy: Strategy,
    train_bars: int,
    test_bars: int,
    cost: CostModel | None = None,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.01,
    overlay: CellPolicy | None = None,
    interval: str = "1m",
    is_maker: bool = False,
    collect_oos_returns: bool = False,
) -> WalkForwardResult:
    """Walk-forward: параметри фіксовані (або оптимізовані вручну зовні),
    стратегія оцінюється на кожному OOS вікні.

    collect_oos_returns: зібрати конкатеновані OOS бар-дохідності у
    результат (для DSR/PBO на OOS без IS-забруднення).
    """
    if len(df) < train_bars + test_bars:
        raise ValueError(f"Дані ({len(df)}) коротші за train+test ({train_bars + test_bars})")

    windows: list[WalkForwardWindow] = []
    oos_ret_parts: list[pd.Series] = []
    idx = 0
    start = 0
    while start + train_bars + test_bars <= len(df):
        tr = df.iloc[start : start + train_bars]
        te = df.iloc[start + train_bars : start + train_bars + test_bars]
        # trades — це потік aggTrades (індекс = час трейду), НЕ бари: ріжемо
        # за ЧАСОМ вікна, а не позиційним зсувом барів (раніше trades.iloc[
        # start:...] давав невірні часові вікна для needs_trades стратегій).
        trades_tr = _slice_by_time(trades, tr.index[0], tr.index[-1]) if trades is not None else None
        trades_te = _slice_by_time(trades, te.index[0], te.index[-1]) if trades is not None else None
        # funding має ВЛАСНИЙ (рідкісний) індекс — ріжемо за часом, не за позицією
        funding_tr = _slice_by_time(funding, tr.index[0], tr.index[-1]) if funding is not None else None
        funding_te = _slice_by_time(funding, te.index[0], te.index[-1]) if funding is not None else None

        res_is = run_backtest(
            tr,
            strategy,
            initial_capital,
            cost,
            position_pct,
            trades_tr,
            funding_tr,
            is_maker=is_maker,
            overlay=overlay,
            interval=interval,
        )
        res_oos = run_backtest(
            te,
            strategy,
            initial_capital,
            cost,
            position_pct,
            trades_te,
            funding_te,
            is_maker=is_maker,
            overlay=overlay,
            interval=interval,
        )

        windows.append(
            WalkForwardWindow(
                window_idx=idx,
                train_start=start,
                train_end=start + train_bars,
                test_start=start + train_bars,
                test_end=start + train_bars + test_bars,
                is_sharpe=_sharpe_from_equity(res_is.equity),
                oos_sharpe=_sharpe_from_equity(res_oos.equity),
                oos_return=float(res_oos.equity.iloc[-1] / res_oos.equity.iloc[0] - 1),
                n_trades=res_oos.metrics.n_trades,
            )
        )
        if collect_oos_returns:
            oos_ret_parts.append(res_oos.equity.pct_change().fillna(0.0))
        idx += 1
        start += test_bars  # крок = розмір OOS (non-overlapping)

    if not windows:
        raise ValueError("Жодного повного вікна — збільшіть train_bars/test_bars або обсяг даних")

    avg_is = float(np.mean([w.is_sharpe for w in windows]))
    avg_oos = float(np.mean([w.oos_sharpe for w in windows]))
    positive = float(np.mean([w.oos_sharpe > 0 for w in windows]))
    degradation = (avg_is - avg_oos) / abs(avg_is) if avg_is != 0 else 0.0

    return WalkForwardResult(
        windows=windows,
        avg_is_sharpe=avg_is,
        avg_oos_sharpe=avg_oos,
        positive_windows_frac=positive,
        degradation=degradation,
        details={"train_bars": train_bars, "test_bars": test_bars},
        oos_returns=pd.concat(oos_ret_parts).sort_index() if collect_oos_returns else None,
    )
