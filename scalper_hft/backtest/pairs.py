"""Парний бектест: статистичний арбітраж двох перп-ног (BTC/ETH/SOL).

Модель:
    - ratio = log(leg1 / leg2); спред PnL = −pos × Δratio (pos=+1: шорт leg1/лонг leg2);
    - funding обох ніг: кожна нога має власну ставку, платіж 1 раз/8h
      (шорт ноги з позитивним фандінгом отримує);
    - комісії: turnover × 2 ноги × fee_rate (taker або maker);
    - ноціонал кожної ноги = position_pct × capital.

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
class PairsResult:
    equity: pd.Series
    positions: pd.Series
    spread: pd.Series
    funding_pnl: float
    metrics: BacktestMetrics
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Pairs arb: ret={m.total_return:+.3%} | SRh={m.sharpe_hourly:+.3f} | угод={m.n_trades} "
            f"| maxDD={m.max_drawdown:.2%} | funding={self.funding_pnl:+.3f}%\n" + m.summary()
        )


def _align(leg1: pd.DataFrame, leg2: pd.DataFrame) -> pd.DataFrame:
    frames = [
        leg1[["close"]].rename(columns={"close": "leg1"}),
        leg2[["close"]].rename(columns={"close": "leg2"}),
    ]
    if {"high", "low"}.issubset(leg1.columns) and {"high", "low"}.issubset(leg2.columns):
        frames.extend(
            [
                leg1[["high", "low"]].add_prefix("l1_"),
                leg2[["high", "low"]].add_prefix("l2_"),
            ]
        )
    return frames[0].join(frames[1:], how="inner").dropna()


def _extract_leg_df(df: pd.DataFrame, leg_num: int) -> pd.DataFrame:
    close_col = f"leg{leg_num}"
    high_col = f"l{leg_num}_high"
    low_col = f"l{leg_num}_low"
    res = pd.DataFrame({"close": df[close_col]}, index=df.index)
    if high_col in df.columns and low_col in df.columns:
        res["high"] = df[high_col]
        res["low"] = df[low_col]
    return res



def _maker_pair_positions(
    signals: pd.Series,
    common: pd.DataFrame,
    position_pct: float,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Та сама модель філу, що й paper: touch + P(fill | distance-to-mid)."""
    from scalper_hft.live.fills import both_or_neither, decide_fill

    target = signals.astype(float).shift(1).fillna(0.0).clip(-1, 1) * position_pct
    actual = np.zeros(len(common))
    curr = 0.0
    c1, c2 = common["leg1"].values, common["leg2"].values
    h1, lo1 = common["l1_high"].values, common["l1_low"].values
    h2, lo2 = common["l2_high"].values, common["l2_low"].values
    for i in range(1, len(common)):
        t = float(target.iloc[i])
        if t == curr:
            actual[i] = curr
            continue
        lim1, lim2 = float(c1[i - 1]), float(c2[i - 1])
        if t > 0:
            s1, s2 = "sell", "buy"
        elif t < 0:
            s1, s2 = "buy", "sell"
        else:
            s1, s2 = ("buy", "sell") if curr > 0 else ("sell", "buy")
        mid1 = 0.5 * (float(h1[i]) + float(lo1[i]))
        mid2 = 0.5 * (float(h2[i]) + float(lo2[i]))
        d1 = decide_fill(s1, lim1, float(h1[i]), float(lo1[i]), mid=mid1, rng=rng)
        d2 = decide_fill(s2, lim2, float(h2[i]), float(lo2[i]), mid=mid2, rng=rng)
        d1, d2 = both_or_neither(d1, d2)
        if d1.filled and d2.filled:
            curr = t
        actual[i] = curr
    return pd.Series(actual, index=common.index)


def run_pairs_backtest(
    leg1: pd.DataFrame,
    leg2: pd.DataFrame,
    strategy: Strategy,
    funding1: pd.DataFrame | None = None,
    funding2: pd.DataFrame | None = None,
    position_pct: float = 0.1,
    cost: CostModel | None = None,
    initial_capital: float = 10_000.0,
    maker_execution: bool = False,
) -> PairsResult:
    """Бектест пари перп-ф'ючерсів на спільному часовому індексі.

    leg1/leg2: klines DataFrame (close). funding1/funding2: funding кожного
    символу (DataFrame 'fundingRate').
    """
    cost = cost or CostModel()
    common = _align(leg1, leg2)
    if len(common) < 200:
        raise ValueError("Замало спільних барів для пари")

    ratio = np.log(common["leg1"] / common["leg2"])
    signals = strategy.generate_signals(common)
    has_range = {"l1_high", "l1_low", "l2_high", "l2_low"}.issubset(common.columns)
    if maker_execution and has_range:
        pos = _maker_pair_positions(signals, common, position_pct)
    else:
        pos = signals.astype(float).shift(1).fillna(0.0).clip(-1, 1) * position_pct

    # спред PnL: pos=+1 (шорт leg1/лонг leg2) → −Δratio
    d_ratio = ratio.diff().fillna(0.0)
    spread_pnl = -pos * d_ratio

    # funding обох ніг (кожна ставка своєї ноги, 1 раз/8h)
    # pos=+1 → leg1 коротка (−pos), leg2 довга (+pos)
    funding_impact = pd.Series(0.0, index=common.index)
    if funding1 is not None and not funding1.empty:
        funding_impact = funding_impact + _leg_funding(-pos, funding1, common)
    if funding2 is not None and not funding2.empty:
        funding_impact = funding_impact + _leg_funding(pos, funding2, common)

    # комісії: turnover × 2 ноги
    leg_cost = cost.maker_cost_per_side() if maker_execution else cost.taker_cost_per_side()
    turnover = (pos - pos.shift(1)).abs().fillna(pos.abs())
    fees = turnover * 2 * leg_cost

    strat_ret = spread_pnl + funding_impact - fees
    equity = (1.0 + strat_ret).cumprod() * initial_capital

    trades = _extract_trades(pos, strat_ret)
    metrics = compute_metrics(equity, trades=trades, exposure=float((pos != 0).mean()), turnover=float(turnover.sum()))
    return PairsResult(
        equity=equity,
        positions=pos,
        spread=ratio,
        funding_pnl=float(funding_impact.sum() * 100),
        metrics=metrics,
        params={"strategy": strategy.name, "position_pct": position_pct},
    )


def _leg_funding(actual_pos: pd.Series, funding: pd.DataFrame, common: pd.DataFrame) -> pd.Series:
    """Funding-вплив однієї ноги: −actual_pos × rate (шорт отримує позитивний фандінг).

    actual_pos — ФАКТИЧНА позиція ноги на спільному індексі (вже зсунута на 1 бар).
    """
    rates = funding["fundingRate"].sort_index()
    bar_idx = common.index.searchsorted(rates.index, side="right") - 1
    mask = (bar_idx >= 0) & (bar_idx < len(common))
    impact = pd.Series(0.0, index=common.index)
    if mask.any():
        valid_bars = bar_idx[mask]
        impact.iloc[valid_bars] = -actual_pos.iloc[valid_bars].values * rates.values[mask]
    return impact


def _extract_trades(pos: pd.Series, strat_ret: pd.Series) -> pd.DataFrame:
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


def run_pairs_walk_forward(
    leg1: pd.DataFrame,
    leg2: pd.DataFrame,
    strategy: Strategy,
    funding1: pd.DataFrame | None = None,
    funding2: pd.DataFrame | None = None,
    train_bars: int = 1500,
    test_bars: int = 500,
    position_pct: float = 0.3,
    maker_execution: bool = True,
) -> dict:
    """Walk-forward для пар: середній OOS SRh по ковзних вікнах."""
    common = _align(leg1, leg2)
    if len(common) < train_bars + test_bars:
        raise ValueError("Замало даних для walk-forward пари")

    oos, is_s = [], []
    start = 0
    while start + train_bars + test_bars <= len(common):
        tr = common.iloc[start : start + train_bars]
        te = common.iloc[start + train_bars : start + train_bars + test_bars]
        l1_tr, l2_tr = _extract_leg_df(tr, 1), _extract_leg_df(tr, 2)
        l1_te, l2_te = _extract_leg_df(te, 1), _extract_leg_df(te, 2)
        f1_tr = _slice_by_time(funding1, tr.index[0], tr.index[-1])
        f2_tr = _slice_by_time(funding2, tr.index[0], tr.index[-1])
        f1_te = _slice_by_time(funding1, te.index[0], te.index[-1])
        f2_te = _slice_by_time(funding2, te.index[0], te.index[-1])
        try:
            r_is = run_pairs_backtest(
                l1_tr, l2_tr, strategy, f1_tr, f2_tr, position_pct=position_pct, maker_execution=maker_execution
            )
            r_oos = run_pairs_backtest(
                l1_te, l2_te, strategy, f1_te, f2_te, position_pct=position_pct, maker_execution=maker_execution
            )
            is_s.append(r_is.metrics.sharpe_hourly)
            oos.append(r_oos.metrics.sharpe_hourly)
        except Exception:  # noqa: BLE001
            oos.append(0.0)
        start += test_bars

    return {
        "avg_is_sharpe": float(np.mean(is_s)) if is_s else 0.0,
        "avg_oos_sharpe": float(np.mean(oos)) if oos else 0.0,
        "positive_windows": float(np.mean([s > 0 for s in oos])) if oos else 0.0,
        "n_windows": len(oos),
    }


def _slice_by_time(funding: pd.DataFrame | None, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame | None:
    if funding is None or funding.empty:
        return None
    mask = (funding.index >= t0) & (funding.index <= t1)
    return funding[mask]
