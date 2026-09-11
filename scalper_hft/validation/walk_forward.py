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

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel

# Роутер (а не run_backtest напряму): market_maker має аудитуватися на
# подієвому рушії, інакше WF оцінює його як нуль-сигнальний векторний прогін.
from scalper_hft.backtest.router import EVENT_STRATEGIES, run_strategy_backtest
from scalper_hft.strategies.base import MissingDataError, Strategy

if TYPE_CHECKING:
    from scalper_hft.overlay.policy import CellPolicy

# Потік (trades/funding) має перекривати щонайменше цю частку періоду даних,
# інакше стратегія тихо повертає нулі на більшій частині вікон.
MIN_STREAM_COVERAGE = 0.5


def _stream_coverage(stream: pd.DataFrame | None, df: pd.DataFrame) -> float:
    """Частка періоду `df`, яку перекриває `stream` за часом."""
    if stream is None or stream.empty or len(df) < 2:
        return 0.0
    lo, hi = df.index[0], df.index[-1]
    span = hi - lo
    if span <= pd.Timedelta(0):
        return 1.0
    covered = min(stream.index[-1], hi) - max(stream.index[0], lo)
    if covered <= pd.Timedelta(0):
        return 0.0
    return float(covered / span)


def require_stream_coverage(
    strategy: Strategy,
    df: pd.DataFrame,
    trades: pd.DataFrame | None,
    funding: pd.DataFrame | None,
    *,
    min_coverage: float = MIN_STREAM_COVERAGE,
) -> None:
    """Fail-fast: заявлені потоки даних мусять перекривати період бектесту.

    Навіщо (аудит 2026-09-11, знахідка K2). `Strategy.validate_inputs` бачить
    ПОВНИЙ переданий фрейм, тому перевірка «trades непорожній» проходить навіть
    тоді, коли aggTrades покривають лише 2 доби з 3 років. Далі walk-forward
    ріже trades за часом для кожного вікна, отримує порожні зрізи, і стратегія
    мовчки повертає суцільні нулі — у `results/iter7_regime_rating.csv` так
    з'явилося 20 клітинок зі `status="ok"` і Sharpe рівно 0.0.

    Тут така ситуація стає помилкою з конкретними числами замість тихого нуля.
    """
    checks = (
        ("trades", trades, getattr(strategy, "needs_trades", False)),
        ("funding", funding, getattr(strategy, "needs_funding", False)),
    )
    for name, stream, needed in checks:
        if not needed or stream is None or stream.empty:
            continue
        coverage = _stream_coverage(stream, df)
        if coverage < min_coverage:
            raise MissingDataError(
                f"{strategy.name}: потік '{name}' перекриває лише {coverage:.1%} періоду даних "
                f"({stream.index[0]} … {stream.index[-1]} проти {df.index[0]} … {df.index[-1]}); "
                f"потрібно ≥{min_coverage:.0%}. Інакше walk-forward отримає порожні вікна і "
                f"стратегія тихо повертатиме нулі. Завантажте дані за весь період."
            )


def _generate_signals(
    strategy: Strategy,
    df: pd.DataFrame,
    trades,
    funding,
    basket_df=None,
    *,
    strict_data: bool = True,
) -> pd.Series:
    """Виклик generate_signals з kwargs, які стратегія реально приймає."""
    if strict_data:
        validate = getattr(strategy, "validate_inputs", None)
        if callable(validate):
            validate(df, trades=trades, funding=funding, basket_df=basket_df)
    params = inspect.signature(strategy.generate_signals).parameters
    kwargs: dict = {}
    if "trades" in params:
        kwargs["trades"] = trades
    if "funding" in params:
        kwargs["funding"] = funding
    if "basket_df" in params:
        kwargs["basket_df"] = basket_df
    return strategy.generate_signals(df, **kwargs)


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
    purge_bars: int = 0,
    embargo_bars: int = 0,
    strict_data: bool = True,
    min_stream_coverage: float = MIN_STREAM_COVERAGE,
) -> WalkForwardResult:
    """Walk-forward: параметри фіксовані (або оптимізовані вручну зовні),
    стратегія оцінюється на кожному OOS вікні.

    collect_oos_returns: зібрати конкатеновані OOS бар-дохідності у
    результат (для DSR/PBO на OOS без IS-забруднення).

    purge_bars: «прогін» між кінцем train і початком test (AFML Ch.7/11) —
        щоб лейбли/позиції, що перетинають межу, не змішували IS і OOS.
        0 = без прогіну (зворотна сумісність).
    embargo_bars: «ембарго» після кожного OOS-вікна перед наступним —
        щоб позиція з попереднього test не «захлювала» наступне train/test.
        0 = без ембарго (зворотна сумісність).
    """
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge_bars/embargo_bars мають бути ≥ 0")
    if len(df) < train_bars + purge_bars + test_bars:
        raise ValueError(f"Дані ({len(df)}) коротші за train+purge+test ({train_bars + purge_bars + test_bars})")
    if strict_data:
        require_stream_coverage(strategy, df, trades, funding, min_coverage=min_stream_coverage)

    windows: list[WalkForwardWindow] = []
    oos_ret_parts: list[pd.Series] = []
    idx = 0
    start = 0

    precomputed: pd.Series | None = None
    if getattr(strategy, "name", "") not in EVENT_STRATEGIES:
        # Один виклик на повній історії: ML/ensemble не бачать Test=500,
        # індикатори зберігають warmup з train. Causal rolling не бере майбутнє.
        precomputed = _generate_signals(strategy, df, trades, funding, strict_data=strict_data)

    while start + train_bars + purge_bars + test_bars <= len(df):
        train_end = start + train_bars
        test_start = train_end + purge_bars  # прогін між train і test
        test_end = test_start + test_bars
        tr = df.iloc[start:train_end]
        te = df.iloc[test_start:test_end]
        # trades — це потік aggTrades (індекс = час трейду), НЕ бари: ріжемо
        # за ЧАСОМ вікна, а не позиційним зсувом барів (раніше trades.iloc[
        # start:...] давав невірні часові вікна для needs_trades стратегій).
        trades_tr = _slice_by_time(trades, tr.index[0], tr.index[-1]) if trades is not None else None
        trades_te = _slice_by_time(trades, te.index[0], te.index[-1]) if trades is not None else None
        # funding має ВЛАСНИЙ (рідкісний) індекс — ріжемо за часом, не за позицією
        funding_tr = _slice_by_time(funding, tr.index[0], tr.index[-1]) if funding is not None else None
        funding_te = _slice_by_time(funding, te.index[0], te.index[-1]) if funding is not None else None

        res_is = run_strategy_backtest(
            tr,
            strategy,
            initial_capital=initial_capital,
            cost=cost,
            position_pct=position_pct,
            trades=trades_tr,
            funding=funding_tr,
            is_maker=is_maker,
            overlay=overlay,
            interval=interval,
            signals=None if precomputed is None else precomputed.reindex(tr.index),
        )
        res_oos = run_strategy_backtest(
            te,
            strategy,
            initial_capital=initial_capital,
            cost=cost,
            position_pct=position_pct,
            trades=trades_te,
            funding=funding_te,
            is_maker=is_maker,
            overlay=overlay,
            interval=interval,
            signals=None if precomputed is None else precomputed.reindex(te.index),
        )

        windows.append(
            WalkForwardWindow(
                window_idx=idx,
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                is_sharpe=_sharpe_from_equity(res_is.equity),
                oos_sharpe=_sharpe_from_equity(res_oos.equity),
                oos_return=float(res_oos.equity.iloc[-1] / res_oos.equity.iloc[0] - 1),
                n_trades=res_oos.metrics.n_trades,
            )
        )
        if collect_oos_returns:
            oos_ret_parts.append(res_oos.equity.pct_change().fillna(0.0))
        idx += 1
        start += test_bars + embargo_bars  # крок = OOS + ембарго (0 = non-overlapping)

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
