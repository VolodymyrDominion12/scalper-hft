"""Сон демона до close 1h-бара: прокинутись за lead_sec до межі свічки."""

from __future__ import annotations

import pandas as pd

from scalper_hft.data.downloader import _interval_ms


def _aware_utc(now: pd.Timestamp | None = None) -> pd.Timestamp:
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def next_bar_close_utc(interval: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Наступна межа інтервалу (якщо now рівно на межі — наступна, не поточна)."""
    now_utc = _aware_utc(now)
    ms = _interval_ms(interval)
    epoch_ms = int(now_utc.timestamp() * 1000)
    rem = epoch_ms % ms
    next_ms = epoch_ms + ms if rem == 0 else epoch_ms - rem + ms
    return pd.Timestamp(next_ms, unit="ms", tz="UTC")


def seconds_until_next_close(interval: str, now: pd.Timestamp | None = None) -> float:
    now_utc = _aware_utc(now)
    nxt = next_bar_close_utc(interval, now_utc)
    return max((nxt - now_utc).total_seconds(), 0.0)


def seconds_until_bar_wake(interval: str, now: pd.Timestamp | None = None, lead_sec: float = 75.0) -> float:
    """Скільки спати після успішного step до наступного вікна обробки.

    За lead_sec до close — прокинутись. Якщо вже в цьому вікні — спати до
    close+2с, щоб closed_klines побачив новий бар.
    """
    to_close = seconds_until_next_close(interval, now)
    if to_close <= lead_sec:
        return max(to_close + 2.0, 1.0)
    return to_close - lead_sec


def daemon_sleep_sec(interval: str, action: str, lead_sec: float = 75.0) -> float:
    """Пауза між кроками демона залежно від останньої дії."""
    if action.startswith("hold:paused"):
        return 5.0
    if action.startswith("hold:same_bar"):
        return max(seconds_until_next_close(interval) + 2.0, 1.0)
    return seconds_until_bar_wake(interval, lead_sec=lead_sec)
