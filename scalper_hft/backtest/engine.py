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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scalper_hft.backtest.execution import CostModel, apply_breakeven_gate
from scalper_hft.backtest.metrics import BacktestMetrics, compute_metrics
from scalper_hft.research.filter_trace import FilterTrace
from scalper_hft.strategies.base import Strategy

if TYPE_CHECKING:
    from scalper_hft.overlay.policy import CellPolicy


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: BacktestMetrics
    params: dict = field(default_factory=dict)
    trace: FilterTrace | None = None  # FilterTrace | None — заповнюється при trace=True

    def summary(self) -> str:
        return self.metrics.summary()

    @property
    def bar_returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def sharpe(self) -> float:
        return self.metrics.sharpe

    @property
    def profit_factor(self) -> float:
        return self.metrics.profit_factor


def _extract_run_trades(pos: pd.Series, strat_ret: pd.Series) -> pd.DataFrame:
    """Угоди з run-ів константної позиції (pairs/delta-neutral семантика).

    ret угоди = Σ strat_ret по барах [entry .. exit] включно при виході у flat
    (exit-комісія бару закриття лишається в угоді), [entry .. exit) при flip;
    відкрита наприкінці — [entry .. останній бар]. Векторизовано.
    """
    cols = ["entry_ts", "exit_ts", "side", "ret"]
    p = pos.to_numpy(dtype=float)
    n = len(p)
    if n == 0 or not np.any(p != 0):
        return pd.DataFrame(columns=cols)
    sr = strat_ret.to_numpy(dtype=float)
    cum = np.concatenate(([0.0], np.cumsum(sr)))

    change = np.empty(n, dtype=bool)
    change[0] = p[0] != 0.0
    change[1:] = p[1:] != p[:-1]
    starts = np.flatnonzero(change)

    rows: list[dict] = []
    for i, s in enumerate(starts):
        v = p[s]
        if v == 0.0:
            continue
        if i + 1 < len(starts):
            x = int(starts[i + 1])
            exit_ts = pos.index[x]
            end_incl = x if p[x] == 0.0 else x - 1  # flat: бар x у угоді; flip: ні
        else:
            exit_ts = pos.index[-1]
            end_incl = n - 1
        rows.append(
            {
                "entry_ts": pos.index[s],
                "exit_ts": exit_ts,
                "side": int(np.sign(v)),
                "ret": float(cum[end_incl + 1] - cum[s]),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _simulate_maker_fills(
    target_vals: np.ndarray,
    close_vals: np.ndarray,
    low_vals: np.ndarray,
    high_vals: np.ndarray,
    *,
    adverse_bps: float = 0.0001,
    prob_touch: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Симуляція Queue Position та Adverse Selection для maker-ордерів.

    Модель "chase": поки цільова позиція не заповнена, ліміт переставляється
    на close попереднього бару. Філ на барі i, якщо low[i] < close[i-1]
    (buy) / high[i] > close[i-1] (sell); рівність — з імовірністю prob_touch.

    Векторизовано по сегментах константного target (parity з колишнім
    Python-циклом — той самий rng і ті самі умови по бару).

    Повертає (actual_pos, adverse_penalties).
    """
    n = len(target_vals)
    actual = np.zeros(n)
    adverse = np.zeros(n)
    if n < 2:
        if n == 1:
            actual[0] = 0.0
        return actual, actual.copy()

    rng = np.random.default_rng(seed)  # відтворюваність бектестів
    rands = rng.random(n)

    close_prev = np.empty(n)
    close_prev[0] = np.nan
    close_prev[1:] = close_vals[:-1]
    buy_fill = (low_vals < close_prev) | ((low_vals == close_prev) & (rands < prob_touch))
    sell_fill = (high_vals > close_prev) | ((high_vals == close_prev) & (rands < prob_touch))
    buy_fill[0] = False
    sell_fill[0] = False

    # сегменти константного target
    seg_change = np.empty(n, dtype=bool)
    seg_change[0] = True
    seg_change[1:] = target_vals[1:] != target_vals[:-1]
    seg_starts = np.flatnonzero(seg_change)
    seg_ends = np.append(seg_starts[1:], n)

    curr = 0.0
    for s, e in zip(seg_starts, seg_ends, strict=True):
        tp = target_vals[s]
        if tp == curr:
            actual[s:e] = curr
            continue
        s0 = max(int(s), 1)  # цикл оригіналу починається з бару 1
        mask = buy_fill[s0:e] if tp > curr else sell_fill[s0:e]
        if mask.any():
            f = s0 + int(np.argmax(mask))  # перший бар філу
            actual[s:f] = curr
            adverse[f] += abs(tp - curr) * adverse_bps
            curr = tp
            actual[f:e] = curr
        else:
            actual[s:e] = curr  # unfilled — позиція лишається
    return actual, adverse


def _extract_trades(positions: pd.Series, ret: pd.Series, fees: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Виділення окремих угод з позиційної серії (вхід/вихід) — векторизовано.

    Угода = максимальний run константної ненульової позиції:
    ціновий PnL = Σ ret×pos по барах run-у; комісія бару зміни позиції
    розщеплюється пропорційно сторонам (exit-частка — закритій позиції,
    entry-частка — новій), як у попередній циклічній версії.

    Окрім часових міток додаються `entry_price`/`exit_price` — ціни виконання
    за моделлю рушія (потрібні для візуалізації точок входу/виходу).
    """
    cols = ["entry_ts", "exit_ts", "side", "ret", "entry_price", "exit_price"]
    p = positions.to_numpy(dtype=float)
    n = len(p)
    if n == 0 or not np.any(p != 0):
        return pd.DataFrame(columns=cols)

    r = ret.to_numpy(dtype=float)
    f = fees.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    prev_c = np.empty(n)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    # межі run-ів (бари, де позиція змінюється)
    change = np.empty(n, dtype=bool)
    change[0] = p[0] != 0.0
    change[1:] = p[1:] != p[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)  # кінець run-у = початок наступного (exclusive)

    turnover = np.abs(np.diff(p, prepend=0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_eff = np.where(turnover > 0, f / turnover, 0.0)

    pnl_bar = p * r
    # префіксні суми для O(1) сум по run-ах
    pnl_cum = np.concatenate(([0.0], np.cumsum(pnl_bar)))

    rows: list[dict] = []
    for i, s in enumerate(starts):
        pos_v = p[s]
        if pos_v == 0.0:
            continue
        e = int(ends[i])
        prev_pos = p[s - 1] if s > 0 else 0.0
        # entry-частка комісії: fee бару входу мінус exit-частка попередньої
        entry_fee = f[s] - rate_eff[s] * abs(prev_pos)
        price_pnl = pnl_cum[e] - pnl_cum[s]
        if i + 1 < len(starts):
            x = int(starts[i + 1])  # бар виходу = перший бар наступного run-у
            exit_fee = rate_eff[x] * abs(pos_v)
            exit_ts = positions.index[x]
            trade_ret = price_pnl - entry_fee - exit_fee
        else:
            # позиція лишилась відкритою до кінця — без exit-комісії
            x = n - 1
            exit_ts = positions.index[-1]
            trade_ret = price_pnl - entry_fee
        rows.append(
            {
                "entry_ts": positions.index[s],
                "exit_ts": exit_ts,
                "side": int(np.sign(pos_v)),
                "ret": float(trade_ret),
                "entry_price": float(prev_c[s]) if np.isfinite(prev_c[s]) else float(c[s]),
                "exit_price": float(prev_c[x]) if np.isfinite(prev_c[x]) else float(c[x]),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _attach_exit_levels(trades: pd.DataFrame, levels: pd.DataFrame | None) -> pd.DataFrame:
    """Прикріпити рівні SL/TP (ціни) до угод.

    Рівні беруться на барі ВХОДУ угоди, за її стороною (лонг/шорт).
    levels: DataFrame з колонками sl_long/tp_long/sl_short/tp_short,
    індексований як df — зі стратегії через Strategy.exit_levels().
    """
    if levels is None or trades is None or trades.empty:
        return trades
    t = trades.copy()
    lv = levels.reindex(t["entry_ts"])
    t["sl_price"] = np.where(t["side"] == 1, lv["sl_long"], lv["sl_short"]).astype(float)
    t["tp_price"] = np.where(t["side"] == 1, lv["tp_long"], lv["tp_short"]).astype(float)
    return t


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10_000.0,
    cost: CostModel | None = None,
    position_pct: float = 0.01,
    trades: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    is_maker: bool = False,
    trace: bool = False,
    overlay: CellPolicy | None = None,
    interval: str = "1m",
) -> BacktestResult:
    """Запуск бектесту стратегії на свічкових даних.

    df: DataFrame з колонками open/high/low/close/volume.
    strategy: екземпляр Strategy (generate_signals(df, trades, funding)).
    trades: aggTrades DataFrame для стратегій, що потребують потоку заявок.
    funding: DataFrame з 'fundingRate' (індекс — час ставки). Додає funding
        грошовий потік: лонг платить позитивний фандінг, шорт отримує.
    is_maker: якщо True — використання maker-комісії (лімітні ордери).
    trace: якщо True — записує FilterTrace (трейс фільтрів) в result.trace.
    """
    if len(df) < 30:
        raise ValueError("Замало даних для бектесту")
    cost = cost or CostModel()
    filter_trace = None
    if trace:
        signals, filter_trace = strategy.generate_signals_traced(df, trades=trades, funding=funding)
    else:
        # Передаємо ОБИДВА потоки, якщо стратегія їх потребує (ensemble/
        # supervisor з mixed-дітьми): раніше needs_trades блокував funding.
        kwargs: dict = {}
        if getattr(strategy, "needs_trades", False):
            kwargs["trades"] = trades
        if getattr(strategy, "needs_funding", False):
            kwargs["funding"] = funding
        signals = strategy.generate_signals(df, **kwargs)
    if len(signals) != len(df):
        raise ValueError("Довжина сигналів не збігається з даними")

    use_gate = bool(getattr(strategy, "use_breakeven_gate", False))
    if overlay is not None:
        from scalper_hft.overlay.apply import apply_cell_overlay

        signals = apply_cell_overlay(signals, overlay, interval=interval, funding=funding)
        position_pct = position_pct * overlay.size_mult
        is_maker = overlay.execution == "maker"
        use_gate = use_gate or overlay.cost_gate

    # Breakeven-гейт: не торгуємо, якщо очікуваний рух < round-trip витрат
    if use_gate:
        signals = apply_breakeven_gate(signals, df, cost, is_maker=is_maker)

    close = df["close"]
    ret = close.pct_change().fillna(0.0)

    # Вектор цільових позицій (з лагом 1)
    target_pos = signals.astype(float).shift(1).fillna(0.0).clip(-1, 1) * position_pct

    if is_maker:
        # Симуляція Queue Position та Adverse Selection для Maker-ордерів
        actual_pos, adverse_penalties = _simulate_maker_fills(
            target_pos.to_numpy(dtype=float),
            close.to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["high"].to_numpy(dtype=float),
        )
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
    # Якщо бар грубіший за каденцію ставок (напр. 1d-бар і 3 ставки/день) —
    # ставки групуються за баром і СУМУЮТЬСЯ (раніше лишалась лише остання).
    if funding is not None and not funding.empty:
        rates = funding["fundingRate"].sort_index()
        bar_idx = df.index.searchsorted(rates.index, side="right") - 1  # останній бар ≤ fts
        mask = (bar_idx >= 0) & (bar_idx < len(df))
        if mask.any():
            bars = pd.Index(bar_idx[mask])
            summed = pd.Series(rates.values[mask], index=bars).groupby(level=0).sum()
            funding_impact = pd.Series(0.0, index=df.index)
            funding_impact.iloc[summed.index.values] = -(pos.iloc[summed.index.values].values * summed.values)
            strat_ret = strat_ret + funding_impact

    equity = (1.0 + strat_ret).cumprod() * initial_capital

    trades_df = _extract_trades(pos, ret, fees, close)
    exit_levels = getattr(strategy, "exit_levels", None)
    trades_df = _attach_exit_levels(trades_df, exit_levels(df) if exit_levels is not None else None)
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
        params={"strategy": strategy.name, "position_pct": position_pct, "is_maker": is_maker, "interval": interval},
        trace=filter_trace,
    )
