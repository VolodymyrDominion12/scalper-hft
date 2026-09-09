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

from scalper_hft.backtest.execution import CostModel, _atr_from_ohlc, apply_breakeven_gate
from scalper_hft.backtest.metrics import BacktestMetrics, compute_metrics
from scalper_hft.backtest.micro_price import QueuePositionModel
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
    def returns(self) -> pd.Series:
        """Барні прибутковості (аліас до bar_returns)."""
        return self.bar_returns

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


def _queue_fill_features(
    df: pd.DataFrame,
    target_pos: pd.Series,
    queue_model: QueuePositionModel,
    *,
    trades: pd.DataFrame | None = None,
    spread_bps: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Побарові (fill_prob, adverse_risk) для maker-симуляції з QueuePositionModel.

    Барові проксі (без L2-історії):
        - distance_bps = spread_bps / 2 (пасивний ліміт на half-spread);
        - vol_frac = ATR14/close;
        - vpin та imbalance — з тікового потоку (buy_ratio по барах), інакше
          нейтральні 0.5 / 0.0;
        - side — напрямок зміни цільової позиції (яку ногу ми ловимо).
    """
    n = len(df)
    close = df["close"]
    vol_frac = (_atr_from_ohlc(df) / close.replace(0, np.nan)).fillna(0.0).to_numpy(dtype=float)

    imbalance: np.ndarray = np.zeros(n)
    vpin: np.ndarray = np.full(n, 0.5)
    if trades is not None and not trades.empty and "side" in trades.columns:
        from scalper_hft.features.indicators import _infer_resample, cvd_from_trades

        try:
            cvd = cvd_from_trades(trades, resample=_infer_resample(df))
            br_series = cvd["buy_ratio"].reindex(df.index).ffill()
            br: np.ndarray = br_series.clip(0.0, 1.0).fillna(0.5).to_numpy(dtype=float)
            imbalance = 2.0 * br - 1.0
            vpin = np.abs(imbalance)
        except (ValueError, KeyError):
            pass

    t = target_pos.to_numpy(dtype=float)
    dside = np.sign(np.diff(t, prepend=0.0))
    # несемо останній ненульовий бік уперед (segment-continuation)
    for i in range(1, n):
        if dside[i] == 0.0:
            dside[i] = dside[i - 1]

    distance = spread_bps / 2.0
    fill_prob = np.empty(n)
    adverse_risk = np.empty(n)
    for i in range(n):
        fill_prob[i] = queue_model.estimate_fill_prob(
            distance_bps=distance,
            queue_ahead_ratio=0.5,
            vol_frac=float(vol_frac[i]),
            vpin=float(vpin[i]),
        )
        adverse_risk[i] = queue_model.adverse_selection_risk(
            side=int(dside[i]) if dside[i] != 0 else 1,
            imbalance=float(imbalance[i]),
            vpin=float(vpin[i]),
        )
    return fill_prob, adverse_risk


def _simulate_maker_fills(
    target_vals: np.ndarray,
    close_vals: np.ndarray,
    low_vals: np.ndarray,
    high_vals: np.ndarray,
    *,
    adverse_bps: float = 0.0001,
    prob_touch: float = 0.5,
    seed: int = 42,
    fill_prob: np.ndarray | None = None,
    adverse_risk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Симуляція Queue Position та Adverse Selection для maker-ордерів.

    Модель "chase": поки цільова позиція не заповнена, ліміт переставляється
    на close попереднього бару. Філ на барі i, якщо low[i] < close[i-1]
    (buy) / high[i] > close[i-1] (sell); рівність — з імовірністю prob_touch.

    Якщо передано `fill_prob` (побарова ймовірність філу з QueuePositionModel),
    то навіть проходження ціни КРІЗЬ рівень не гарантує філ — черга попереду
    може не дійти до нас: fill = touch-through з імовірністю fill_prob[i],
    exact touch — з fill_prob[i] × prob_touch. `adverse_risk` (0..1) масштабує
    adverse-penalty філу (токсичний потік → гірша ціна відносно рішення).

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
    if fill_prob is not None:
        p = np.clip(fill_prob, 0.0, 1.0)
        buy_fill = ((low_vals < close_prev) & (rands < p)) | ((low_vals == close_prev) & (rands < p * prob_touch))
        sell_fill = ((high_vals > close_prev) & (rands < p)) | ((high_vals == close_prev) & (rands < p * prob_touch))
    else:
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
            risk_mult = 1.0 + float(adverse_risk[f]) if adverse_risk is not None else 1.0
            adverse[f] += abs(tp - curr) * adverse_bps * risk_mult
            curr = tp
            actual[f:e] = curr
        else:
            actual[s:e] = curr  # unfilled — позиція лишається
    return actual, adverse


def _apply_intrabar_exits(
    pos: np.ndarray,
    close_vals: np.ndarray,
    high_vals: np.ndarray,
    low_vals: np.ndarray,
    levels: pd.DataFrame,
    index: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    """Внутрішньобарові виходи за SL/TP рівнями стратегії (без lookahead).

    Для кожного run-у константної позиції беруться рівні з бару РІШЕННЯ
    (entry-1 — позиція активна з бару entry через shift(1) лаг). Далі по
    барах run-у: якщо low/high торкається SL/TP — вихід за ЦІНОЮ РІВНЯ на
    тому ж барі; при одночасному дотику SL і TP на одному барі — песимістично
    SL. Позиція після бару виходу обнуляється до кінця run-у (повторний вхід
    — лише за новим сигналом, тобто на межі сегмента target).

    Повертає (pos_adj, ret_override): ret_override[i] не-NaN лише на барі
    стоп-виходу — підміняє close-to-close дохідність бару фактичною
    дохідністю до ціни рівня.
    """
    n = len(pos)
    ret_override = np.full(n, np.nan)
    if n == 0 or not np.any(pos != 0):
        return pos, ret_override

    lv = levels.reindex(index)
    sl_long = lv["sl_long"].to_numpy(dtype=float)
    tp_long = lv["tp_long"].to_numpy(dtype=float)
    sl_short = lv["sl_short"].to_numpy(dtype=float)
    tp_short = lv["tp_short"].to_numpy(dtype=float)

    out = pos.copy()
    change = np.empty(n, dtype=bool)
    change[0] = pos[0] != 0.0
    change[1:] = pos[1:] != pos[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)

    for i, s in enumerate(starts):
        v = pos[s]
        if v == 0.0:
            continue
        e = int(ends[i])
        decision_bar = s - 1  # рівні з бару рішення (без lookahead)
        if decision_bar < 0:
            continue
        side = 1 if v > 0 else -1
        sl = sl_long[decision_bar] if side == 1 else sl_short[decision_bar]
        tp = tp_long[decision_bar] if side == 1 else tp_short[decision_bar]
        if not (np.isfinite(sl) and np.isfinite(tp)):
            continue
        for x in range(s, e):
            if side == 1:
                stop_hit = low_vals[x] <= sl
                tp_hit = high_vals[x] >= tp
            else:
                stop_hit = high_vals[x] >= sl
                tp_hit = low_vals[x] <= tp
            if not (stop_hit or tp_hit):
                continue
            # обидва рівні на одному барі — песимістично SL
            exit_price = sl if stop_hit else tp
            prev_close = close_vals[x - 1] if x > 0 else close_vals[x]
            if prev_close > 0:
                ret_override[x] = side * (exit_price / prev_close - 1.0)
            out[x + 1 : e] = 0.0  # flat до кінця run-у (перевхід — новий сигнал)
            break
    return out, ret_override


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
    queue_model: QueuePositionModel | None = None,
    spread_bps: float = 2.0,
    intrabar_exits: bool = False,
    signals: pd.Series | None = None,
    max_leverage: float | None = None,
    vol_target_ann: float | None = None,
    vol_lookback: int = 168,
) -> BacktestResult:
    """Запуск бектесту стратегії на свічкових даних.

    df: DataFrame з колонками open/high/low/close/volume.
    strategy: екземпляр Strategy (generate_signals(df, trades, funding)).
    signals: готові сигнали (індекс = df.index); якщо задано — generate_signals
        не викликається (walk-forward для ML: модель бачить повну історію).
    trades: aggTrades DataFrame для стратегій, що потребують потоку заявок.
    funding: DataFrame з 'fundingRate' (індекс — час ставки). Додає funding
        грошовий потік: лонг платить позитивний фандінг, шорт отримує.
    is_maker: якщо True — використання maker-комісії (лімітні ордери).
    trace: якщо True — записує FilterTrace (трейс фільтрів) в result.trace.
    queue_model: якщо задано — побарова ймовірність maker-філу з черги
        (QueuePositionModel): проходження ціни крізь ліміт НЕ гарантує філ,
        adverse-penalty масштабується токсичністю потоку. Барові проксі:
        vol_frac = ATR14/close, vpin = |imbalance| тікового потоку (якщо
        trades доступні), queue_ahead=0.5.
    spread_bps: оцінка half-spread (bps) як distance для queue_model
        (калібрується з bookTicker через estimate_spread_from_bookticker).
    intrabar_exits: якщо True і стратегія дає exit_levels — виходи за
        SL/TP філимо за ЦІНОЮ РІВНЯ на барі дотику (песимістично SL при
        одночасному дотику), а не close-to-close наступного бару.
    max_leverage: якщо задано — жорсткий кліп |цільова позиція| ≤ max_leverage
        (позиція у частках equity = плече ноціоналу). None = без кліпу
        (зворотна сумісність).
    vol_target_ann: якщо задано — vol-target sizing: цільова позиція
        масштабується на clip(target_ann / realized_ann, 0, 1), де
        realized_ann — річна σ барових дохідностей за останні vol_lookback
        барів. Множник відомий на закритті бару рішення (shift(1) разом із
        сигналом) — без lookahead. None = фіксований position_pct.
    """
    if len(df) < 30:
        raise ValueError("Замало даних для бектесту")
    cost = cost or CostModel()
    filter_trace = None
    if signals is None:
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
    else:
        signals = signals.reindex(df.index).fillna(0.0)
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

    # Vol-target sizing (Phase 5.2): множник відомий на закритті бару t−1
    # (разом із сигналом), застосовується до позиції бару t — без lookahead.
    if vol_target_ann is not None:
        unit = interval[-1]
        num = int(interval[:-1])
        sec_per_bar = num * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        bpy = 365.0 * 86400.0 / float(sec_per_bar)
        realized = ret.rolling(vol_lookback, min_periods=max(20, vol_lookback // 4)).std(ddof=0) * np.sqrt(bpy)
        vol_mult = (float(vol_target_ann) / realized.replace(0.0, np.nan)).clip(0.0, 1.0)
        vol_mult = vol_mult.shift(1).fillna(1.0)
        target_pos = target_pos * vol_mult

    # Жорсткий ліміт плеча (як у live risk_gate): |позиція| ≤ max_leverage.
    if max_leverage is not None:
        target_pos = target_pos.clip(-float(max_leverage), float(max_leverage))

    if is_maker:
        # Симуляція Queue Position та Adverse Selection для Maker-ордерів
        fill_prob = None
        adverse_risk = None
        if queue_model is not None:
            fill_prob, adverse_risk = _queue_fill_features(
                df, target_pos, queue_model, trades=trades, spread_bps=spread_bps
            )
        actual_pos, adverse_penalties = _simulate_maker_fills(
            target_pos.to_numpy(dtype=float),
            close.to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["high"].to_numpy(dtype=float),
            fill_prob=fill_prob,
            adverse_risk=adverse_risk,
        )
        pos = pd.Series(actual_pos, index=df.index)
        adv_penalty_series = pd.Series(adverse_penalties, index=df.index)
    else:
        # Taker - гарантований філл на close
        pos = target_pos
        adv_penalty_series = pd.Series(0.0, index=df.index)

    # Внутрішньобарові SL/TP: вихід за ціною рівня на барі дотику
    ret_override: pd.Series | None = None
    if intrabar_exits:
        exit_levels_fn = getattr(strategy, "exit_levels", None)
        levels = exit_levels_fn(df) if exit_levels_fn is not None else None
        if levels is not None:
            pos_adj, overrides = _apply_intrabar_exits(
                pos.to_numpy(dtype=float),
                close.to_numpy(dtype=float),
                df["high"].to_numpy(dtype=float),
                df["low"].to_numpy(dtype=float),
                levels,
                df.index,
            )
            pos = pd.Series(pos_adj, index=df.index)
            ret_override = pd.Series(overrides, index=df.index)

    turnover = (pos - pos.shift(1)).abs().fillna(pos.abs())
    # Vol-aware slippage (Narang гл. 5): taker slippage масштабується
    # поточною волатильністю, якщо задано cost.vol_ref (як у pairs-шляху).
    fee_rate: float | pd.Series
    if is_maker:
        fee_rate = cost.maker_cost_per_side()
    elif cost.vol_ref > 0:
        vol_frac = (_atr_from_ohlc(df) / close.replace(0, np.nan)).fillna(cost.vol_ref)
        fee_rate = cost.taker_fee + cost.vol_aware_slippage(vol_frac)
    else:
        fee_rate = cost.taker_cost_per_side()
    fees = turnover * fee_rate + adv_penalty_series

    # Прибуток за бар t = позиція, активна в t, × дохідність бару t, мінус комісії.
    bar_ret = ret if ret_override is None else ret_override.where(ret_override.notna(), ret)
    strat_ret = pos * bar_ret - fees

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
