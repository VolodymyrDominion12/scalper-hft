"""Filter Tracing — відстеження заблокованих сигналів.

Мета (книга Narang, гл. 3–5): кожен фільтр у стратегії має покращувати
edge. Якщо фільтр відкидає переважно прибуткові сигнали — він шкодить.
Цей модуль збирає «тіньові» угоди (raw сигнал є, фільтр заблокував)
та атрибутує P&L до кожного фільтру.

Використання в стратегії:
    from scalper_hft.research.filter_trace import FilterTrace, SignalEvent

    def generate_signals_traced(self, df, **kw):
        trace = FilterTrace()
        # ... логіка ...
        for ts, row in df.iterrows():
            raw = compute_raw_signal(row)
            if raw != 0:
                blocked = []
                if not vol_ok[ts]: blocked.append("vol_ok")
                if not trend_ok[ts]: blocked.append("trend_ok")
                trace.add(SignalEvent(ts=ts, raw=raw,
                                     final=0 if blocked else raw,
                                     blocked_by=blocked,
                                     context={"rsi": rsi_val[ts]}))
        signals = pd.Series(...)
        return signals, trace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SignalEvent:
    """Один потенційний сигнал на барі — з результатом та причиною блокування."""

    ts: pd.Timestamp
    raw_signal: int  # +1 / -1 — що хотіла логіка
    final_signal: int  # 0 / +1 / -1 — що пройшло після всіх фільтрів
    blocked_by: list[str]  # список назв фільтрів, що заблокували
    context: dict[str, Any] = field(default_factory=dict)  # rsi, atr, trend, ...

    @property
    def was_blocked(self) -> bool:
        return self.final_signal == 0 and self.raw_signal != 0

    @property
    def side(self) -> str:
        return "long" if self.raw_signal == 1 else "short"


class FilterTrace:
    """Колекція SignalEvent для одного бектесту.

    Збирає події → конвертує у DataFrame → обчислює атрибуцію фільтрів.
    """

    def __init__(self) -> None:
        self._events: list[SignalEvent] = []

    def add(self, event: SignalEvent) -> None:
        self._events.append(event)

    def __len__(self) -> int:
        return len(self._events)

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертує в DataFrame для аналізу та збереження."""
        if not self._events:
            return pd.DataFrame(columns=["ts", "raw_signal", "final_signal", "blocked_by", "was_blocked", "side"])
        rows = []
        for e in self._events:
            row: dict[str, Any] = {
                "ts": e.ts,
                "raw_signal": e.raw_signal,
                "final_signal": e.final_signal,
                "blocked_by": "|".join(e.blocked_by) if e.blocked_by else "",
                "was_blocked": e.was_blocked,
                "side": e.side,
            }
            row.update(e.context)
            rows.append(row)
        return pd.DataFrame(rows).set_index("ts")

    def n_blocked(self) -> int:
        return sum(1 for e in self._events if e.was_blocked)

    def n_passed(self) -> int:
        return sum(1 for e in self._events if not e.was_blocked)

    def blocked_events(self) -> list[SignalEvent]:
        return [e for e in self._events if e.was_blocked]

    def passed_events(self) -> list[SignalEvent]:
        return [e for e in self._events if not e.was_blocked]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame | None) -> FilterTrace:
        """Відновити трейс з таблиці `to_dataframe` (у т.ч. parquet артефактів)."""
        trace = cls()
        if df is None or len(df) == 0:
            return trace
        work = df.copy()
        if "ts" not in work.columns:
            work = work.reset_index()
            if "index" in work.columns and "ts" not in work.columns:
                work = work.rename(columns={"index": "ts"})
        if "ts" not in work.columns:
            return trace
        reserved = frozenset({"ts", "raw_signal", "final_signal", "blocked_by", "was_blocked", "side"})
        for rec in work.to_dict("records"):
            ts_raw = rec.get("ts")
            if ts_raw is None or (isinstance(ts_raw, float) and ts_raw != ts_raw):
                continue
            blocked_raw = rec.get("blocked_by") or ""
            if isinstance(blocked_raw, str):
                blocked = [part for part in blocked_raw.split("|") if part]
            elif isinstance(blocked_raw, (list, tuple)):
                blocked = [str(part) for part in blocked_raw if part]
            else:
                blocked = []
            context = {
                str(key): value
                for key, value in rec.items()
                if key not in reserved and value is not None and not (isinstance(value, float) and value != value)
            }
            trace.add(
                SignalEvent(
                    ts=pd.Timestamp(ts_raw),
                    raw_signal=int(rec.get("raw_signal") or 0),
                    final_signal=int(rec.get("final_signal") or 0),
                    blocked_by=blocked,
                    context=context,
                )
            )
        return trace


def filter_attribution(trace: FilterTrace) -> pd.DataFrame:
    """Атрибуція заблокованих сигналів по фільтрам.

    Повертає DataFrame:
        filter_name | n_blocked | pct_of_all_signals

    Примітка: один сигнал може бути заблокований кількома фільтрами одночасно.
    """
    if not trace._events:
        return pd.DataFrame(columns=["filter_name", "n_blocked", "pct_of_all_signals"])

    total = len(trace._events)
    counter: dict[str, int] = {}
    for e in trace._events:
        for f in e.blocked_by:
            counter[f] = counter.get(f, 0) + 1

    if not counter:
        return pd.DataFrame(columns=["filter_name", "n_blocked", "pct_of_all_signals"])

    rows = [
        {"filter_name": k, "n_blocked": v, "pct_of_all_signals": v / total}
        for k, v in sorted(counter.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


def filter_pnl_impact(
    trace: FilterTrace,
    close: pd.Series,
    horizon_bars: int = 5,
) -> pd.DataFrame:
    """Оцінка потенційного P&L заблокованих сигналів (тіньових угод).

    Метод: для кожного заблокованого сигналу рахує forward return за
    `horizon_bars` барів з моменту сигналу × напрямок сигналу.
    Це «що було б», якби фільтр не спрацював.

    close: Series цін закриття (індекс datetime, як у df бектесту).
    horizon_bars: горизонт утримання для shadow-trade оцінки.

    Повертає DataFrame:
        filter_name | n_blocked | shadow_mean_ret | shadow_win_rate | shadow_total_pnl
    """
    blocked = trace.blocked_events()
    if not blocked:
        return pd.DataFrame(
            columns=["filter_name", "n_blocked", "shadow_mean_ret", "shadow_win_rate", "shadow_total_pnl"]
        )

    # Індексуємо close для швидкого пошуку
    close_arr = close.values
    close_idx = {ts: i for i, ts in enumerate(close.index)}

    # Per-event forward return
    event_results: list[dict[str, Any]] = []
    for e in blocked:
        i = close_idx.get(e.ts)
        if i is None or i + horizon_bars >= len(close_arr):
            continue
        entry_px = close_arr[i]
        exit_px = close_arr[i + horizon_bars]
        if entry_px <= 0:
            continue
        fwd_ret = (exit_px / entry_px - 1.0) * e.raw_signal  # знак за напрямком
        event_results.append({"blocked_by": e.blocked_by, "fwd_ret": fwd_ret})

    if not event_results:
        return pd.DataFrame(
            columns=["filter_name", "n_blocked", "shadow_mean_ret", "shadow_win_rate", "shadow_total_pnl"]
        )

    # Атрибуція по фільтрах
    filter_rets: dict[str, list[float]] = {}
    for r in event_results:
        for f in r["blocked_by"]:
            filter_rets.setdefault(f, []).append(r["fwd_ret"])

    rows = []
    for fname, rets in sorted(filter_rets.items(), key=lambda x: -len(x[1])):
        arr = np.array(rets)
        rows.append(
            {
                "filter_name": fname,
                "n_blocked": len(arr),
                "shadow_mean_ret": float(arr.mean()),
                "shadow_win_rate": float((arr > 0).mean()),
                "shadow_total_pnl": float(arr.sum()),
            }
        )
    return pd.DataFrame(rows)


def compare_with_without_filter(
    trace: FilterTrace,
    actual_equity: pd.Series,
    close: pd.Series,
    initial_capital: float = 10_000.0,
    horizon_bars: int = 5,
) -> dict[str, pd.Series]:
    """Порівняння equity кривих: фактична vs «без фільтрів» (з тіньовими угодами).

    Дає грубу оцінку: яка крива капіталу була б, якби усі заблоковані
    сигнали виконувались (за forward-return оцінкою).

    Повертає dict {"actual": Series, "no_filters": Series}.
    """
    blocked = trace.blocked_events()
    if not blocked:
        return {"actual": actual_equity, "no_filters": actual_equity.copy()}

    close_idx = {ts: i for i, ts in enumerate(close.index)}
    close_arr = close.values

    # Додаткові повернення від тіньових угод
    shadow_ret = pd.Series(0.0, index=actual_equity.index)
    for e in blocked:
        i = close_idx.get(e.ts)
        if i is None or i + horizon_bars >= len(close_arr):
            continue
        entry_px = close_arr[i]
        exit_px = close_arr[i + horizon_bars]
        if entry_px <= 0:
            continue
        fwd_ret = (exit_px / entry_px - 1.0) * e.raw_signal
        # Розподіляємо рівномірно по барах утримання
        per_bar = fwd_ret / horizon_bars
        for k in range(1, horizon_bars + 1):
            if i + k < len(close.index):
                bar_ts = close.index[i + k]
                if bar_ts in shadow_ret.index:
                    shadow_ret[bar_ts] += per_bar

    actual_ret = actual_equity.pct_change().fillna(0.0)
    no_filter_ret = actual_ret + shadow_ret.reindex(actual_ret.index, fill_value=0.0)
    no_filter_equity = (1.0 + no_filter_ret).cumprod() * initial_capital

    return {"actual": actual_equity, "no_filters": no_filter_equity}
