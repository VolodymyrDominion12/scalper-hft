"""Risk gate: cooldown → halt, flatten завжди дозволений.

Порт trade-bot CircuitBreaker (м'який cooldown size×0.5 перед повним halt)
і кап корельованого ноціоналу для пар, що ділять ногу (BTC-кластер).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class CooldownState:
    until: pd.Timestamp | None = None
    reason: str = ""

    def active(self, now: pd.Timestamp) -> bool:
        return self.until is not None and pd.Timestamp(now) < pd.Timestamp(self.until)


@dataclass(frozen=True, slots=True)
class EntryDecision:
    """allow | cooldown | reject. size_mult застосовується лише до нових входів."""

    status: str
    size_mult: float
    reason: str
    cooldown: CooldownState


def decide_entry(
    *,
    consecutive_losses: int,
    now: pd.Timestamp,
    cooldown: CooldownState,
    cooldown_losses: int = 2,
    max_consecutive_losses: int = 3,
    cooldown_hours: float = 12.0,
    cooldown_size_mult: float = 0.5,
    flattening: bool = False,
    calendar_df: pd.DataFrame | None = None,
    news_window_mins: int = 15,
) -> EntryDecision:
    """Дозволити / зменшити / відхилити новий вхід.

    Flatten (закриття) ніколи не блокується. Halt має пріоритет над cooldown.
    Перевіряє також макроекономічний календар (за наявності), щоб блокувати
    входи під час важливих новин.
    """
    if flattening:
        return EntryDecision("allow", 1.0, "flatten", cooldown)

    # Макроекономічний фільтр (блокує нові позиції)
    if calendar_df is not None and not calendar_df.empty:
        from scalper_hft.data.economic_calendar import is_news_time

        if is_news_time(now, calendar_df, window_before_mins=news_window_mins, window_after_mins=news_window_mins):
            return EntryDecision("reject", 0.0, "high_impact_news", cooldown)

    if consecutive_losses >= max_consecutive_losses:
        return EntryDecision("reject", 0.0, "серія збитків — пауза", cooldown)
    if cooldown.active(now):
        return EntryDecision("cooldown", cooldown_size_mult, "cooldown", cooldown)
    if consecutive_losses >= cooldown_losses:
        until = pd.Timestamp(now) + pd.Timedelta(hours=float(cooldown_hours))
        new_state = CooldownState(until=until, reason="consecutive_losses")
        return EntryDecision("cooldown", cooldown_size_mult, "cooldown", new_state)
    return EntryDecision("allow", 1.0, "ok", cooldown)


@dataclass(slots=True)
class DrawdownBreaker:
    """Peak-to-trough circuit breaker (Narang: drawdown control).

    На відміну від daily/weekly лімітів (якір = старт дня/тижня), тут якір —
    ІСТОРИЧНИЙ ПІК equity: просідання від піку > max_dd_pct → halt + flatten.
    Спрацьовування липке (triggered) до явного reset() — не "відлікується"
    відскоком equity.
    """

    max_dd_pct: float
    high_water: float = 0.0
    triggered: bool = False

    def check(self, equity: float) -> bool:
        """Оновити пік і повернути True, якщо breaker спрацьований."""
        if self.triggered:
            return True
        self.high_water = max(self.high_water, float(equity))
        if self.high_water > 0 and equity <= self.high_water * (1.0 - self.max_dd_pct):
            self.triggered = True
        return self.triggered

    def reset(self, equity: float) -> None:
        """Ручне зняття halt (оператор): новий якір = поточний equity."""
        self.triggered = False
        self.high_water = float(equity)


def leverage_ok(
    current_notional: float,
    add_notional: float,
    equity: float,
    max_leverage: float,
) -> bool:
    """Чи проходить сумарний ноціонал ліміт плеча: (cur + add) ≤ equity × max_leverage."""
    if equity <= 0 or max_leverage <= 0:
        return False
    return (float(current_notional) + float(add_notional)) <= equity * max_leverage


def pair_legs(pid: str) -> frozenset[str]:
    clean_pid = pid.rsplit(":", 1)[0] if ":" in pid else pid
    return frozenset(part for part in clean_pid.split("/") if part)


def pairs_share_leg(a: str, b: str) -> bool:
    return bool(pair_legs(a) & pair_legs(b))


def correlated_cluster(pid: str, open_pids: Iterable[str]) -> list[str]:
    return [p for p in open_pids if pairs_share_leg(pid, p)]


def correlated_size_mult(
    new_pid: str,
    new_size_pct: float,
    open_size_pcts: Mapping[str, float],
    cap: float,
) -> float:
    """Множник [0, 1], щоб сума size_pct корельованого кластера ≤ cap.

    Пари без спільної ноги не входять у кластер. Поточна пара (ще не відкрита)
    не повинна бути в open_size_pcts.
    """
    if cap <= 0 or new_size_pct <= 0:
        return 0.0
    cluster = correlated_cluster(new_pid, open_size_pcts)
    existing = sum(float(open_size_pcts[p]) for p in cluster)
    room = float(cap) - existing
    if room <= 0:
        return 0.0
    return float(min(1.0, room / new_size_pct))


def pos_pair_id(position_key: str) -> str:
    """'XRPUSDT/BTCUSDT:XRPUSDT' → 'XRPUSDT/BTCUSDT'."""
    return position_key.rsplit(":", 1)[0] if ":" in position_key else position_key


def open_pair_size_pcts(position_keys: Iterable[str], size_pct: float) -> dict[str, float]:
    """Відкриті пари з акаунта → {pair_id: size_pct} (однаковий size_pct у портфелі)."""
    return {pos_pair_id(key): float(size_pct) for key in position_keys if ":" in key}
