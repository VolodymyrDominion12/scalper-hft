"""Delta-neutral бектест: перп + спот (арбітраж фандінгу).

Модель пари (notional кожної ноги = position_pct × capital):
    - carry = +1: шорт перпа + лонг споту → ціновий PnL = −Δbasis_pct;
    - carry = −1: лонг перпа + шорт споту → ціновий PnL = +Δbasis_pct;
    - basis_pct = perp_close / spot_close − 1 (вирівняно за часом);
    - funding PnL: один раз за 8h-блок = carry × rate × position_pct
      (шорт перпа отримує позитивний фандінг);
    - комісії: turnover × 2 ноги × fee_rate (taker).

Без lookahead: сигнал на закритті t → позиція з бару t+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.backtest.metrics import BacktestMetrics, compute_metrics
from scalper_hft.strategies.base import Strategy


@dataclass
class DeltaNeutralResult:
    equity: pd.Series
    positions: pd.Series
    basis: pd.Series
    funding_pnl: float
    metrics: BacktestMetrics
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Delta-neutral arb: ret={m.total_return:+.3%} | SRh={m.sharpe_hourly:+.3f} "
            f"| угод={m.n_trades} | maxDD={m.max_drawdown:.2%}\n"
            f"  funding PnL: {self.funding_pnl:+.3f}% від капіталу | basis std: {self.basis.std():.4%}\n" + m.summary()
        )


def run_delta_neutral_backtest(
    perp: pd.DataFrame,
    spot: pd.DataFrame,
    strategy: Strategy,
    funding: pd.DataFrame | None = None,
    position_pct: float = 0.1,
    cost: CostModel | None = None,
    initial_capital: float = 10_000.0,
    maker_execution: bool = False,
) -> DeltaNeutralResult:
    """Бектест delta-neutral пари на спільному часовому індексі.

    perp/spot: klines DataFrame (close обов'язкова). Індекси вирівнюються
    через join (inner на спільних барах).
    maker_execution: якщо True — комісії maker на обох ногах (post-only);
    для арбітражу фандінгу критично (taker 2 ніг вбиває прибуток).
    """
    cost = cost or CostModel()
    if funding is None or funding.empty:
        raise ValueError("Delta-neutral arb потребує funding даних")

    common = (
        perp[["close"]]
        .rename(columns={"close": "perp"})
        .join(spot[["close"]].rename(columns={"close": "spot"}), how="inner")
        .dropna()
    )
    if len(common) < 100:
        raise ValueError("Замало спільних барів перп/спот")

    basis = common["perp"] / common["spot"] - 1.0

    # сигнали на загальному індексі (стратегія потребує funding)
    signals = strategy.generate_signals(common, funding=funding)
    pos = signals.astype(float).shift(1).fillna(0.0).clip(-1, 1) * position_pct

    # ціновий PnL пари: carry=+1 (шорт перп) → −Δbasis
    dbasis = basis.diff().fillna(0.0)
    price_pnl = -pos * dbasis

    # funding: один раз за 8h-блок (carry × rate × position_pct).
    # Якщо бар грубіший за каденцію ставок (напр. 1d-бар і 3 ставки/день) —
    # ставки групуються за баром і СУМУЮТЬСЯ (раніше лишалась лише остання).
    rates = funding["fundingRate"].sort_index()
    bar_idx = common.index.searchsorted(rates.index, side="right") - 1
    mask = (bar_idx >= 0) & (bar_idx < len(common))
    funding_impact = pd.Series(0.0, index=common.index)
    if mask.any():
        bars = pd.Index(bar_idx[mask])
        summed = pd.Series(rates.values[mask], index=bars).groupby(level=0).sum()
        funding_impact.iloc[summed.index.values] = pos.iloc[summed.index.values].values * summed.values

    # комісії: turnover × 2 ноги (maker або taker)
    leg_cost = cost.maker_cost_per_side() if maker_execution else cost.taker_cost_per_side()
    turnover = (pos - pos.shift(1)).abs().fillna(pos.abs())
    fees = turnover * 2 * leg_cost

    strat_ret = price_pnl + funding_impact - fees
    equity = (1.0 + strat_ret).cumprod() * initial_capital

    trades = _extract_pair_trades(pos, strat_ret)
    metrics = compute_metrics(equity, trades=trades, exposure=float((pos != 0).mean()), turnover=float(turnover.sum()))
    return DeltaNeutralResult(
        equity=equity,
        positions=pos,
        basis=basis,
        funding_pnl=float(funding_impact.sum() * 100),
        metrics=metrics,
        params={"strategy": strategy.name, "position_pct": position_pct},
    )


def _extract_pair_trades(pos: pd.Series, strat_ret: pd.Series) -> pd.DataFrame:
    """Угоди пари: вхід/вихід carry-позиції."""
    rows: list[dict] = []
    cur = 0.0
    entry_ts = None
    cum = 0.0
    for ts, p in pos.items():
        if p != cur:
            if cur != 0 and entry_ts is not None:
                rows.append(
                    {"entry_ts": entry_ts, "exit_ts": ts, "side": int(cur / abs(cur)) if cur else 0, "ret": cum}
                )
            entry_ts = ts if p != 0 else None
            cum = 0.0
            cur = p
        if cur != 0 and entry_ts is not None:
            cum += strat_ret.get(ts, 0.0)
    if cur != 0 and entry_ts is not None:
        rows.append({"entry_ts": entry_ts, "exit_ts": pos.index[-1], "side": int(cur / abs(cur)), "ret": cum})
    return (
        pd.DataFrame(rows, columns=["entry_ts", "exit_ts", "side", "ret"])
        if rows
        else pd.DataFrame(columns=["entry_ts", "exit_ts", "side", "ret"])
    )


def run_dn_walk_forward(
    perp: pd.DataFrame,
    spot: pd.DataFrame,
    strategy: Strategy,
    funding: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    position_pct: float = 0.1,
) -> dict:
    """Walk-forward для delta-neutral: середній OOS Sharpe по ковзних вікнах.

    Повертає словник з avg_is_sharpe, avg_oos_sharpe, часткою позитивних вікон.
    """
    common = (
        perp[["close"]]
        .rename(columns={"close": "perp"})
        .join(spot[["close"]].rename(columns={"close": "spot"}), how="inner")
        .dropna()
    )
    if len(common) < train_bars + test_bars:
        raise ValueError("Замало даних для walk-forward")

    oos_sharpes: list[float] = []
    is_sharpes: list[float] = []
    start = 0
    while start + train_bars + test_bars <= len(common):
        tr = common.iloc[start : start + train_bars]
        te = common.iloc[start + train_bars : start + train_bars + test_bars]
        perp_tr = tr[["perp"]].rename(columns={"perp": "close"})
        spot_tr = tr[["spot"]].rename(columns={"spot": "close"})
        perp_te = te[["perp"]].rename(columns={"perp": "close"})
        spot_te = te[["spot"]].rename(columns={"spot": "close"})
        f_tr = _slice_funding(funding, tr.index[0], tr.index[-1])
        f_te = _slice_funding(funding, te.index[0], te.index[-1])
        try:
            res_is = run_delta_neutral_backtest(perp_tr, spot_tr, strategy, f_tr, position_pct)
            res_oos = run_delta_neutral_backtest(perp_te, spot_te, strategy, f_te, position_pct)
            is_sharpes.append(res_is.metrics.sharpe_hourly)
            oos_sharpes.append(res_oos.metrics.sharpe_hourly)
        except Exception:  # noqa: BLE001
            oos_sharpes.append(0.0)
        start += test_bars

    return {
        "avg_is_sharpe": float(np.mean(is_sharpes)) if is_sharpes else 0.0,
        "avg_oos_sharpe": float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
        "positive_windows": float(np.mean([s > 0 for s in oos_sharpes])) if oos_sharpes else 0.0,
        "n_windows": len(oos_sharpes),
    }


def _slice_funding(funding: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    mask = (funding.index >= t0) & (funding.index <= t1)
    return funding[mask]
