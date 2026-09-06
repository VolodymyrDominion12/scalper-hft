"""Накласти CellPolicy на Series сигналів без lookahead.

Рішення на барі t використовує лише сигнал ≤ t, стан позиції до t
і funding з індексом ≤ t.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scalper_hft.data.resample import interval_minutes
from scalper_hft.overlay.policy import CellPolicy

_FUNDING_PERIODS_PER_YEAR = 3.0 * 365.0
_FUNDING_LOOKBACK_DAYS = 14


def hours_to_bars(min_hold_hours: float, interval: str) -> int:
    """Мінімум барів, що покриває min_hold_hours на даному ТФ (стела вгору)."""
    if min_hold_hours <= 0:
        return 0
    minutes = interval_minutes(interval)
    if minutes <= 0:
        raise ValueError(f"interval_minutes({interval!r}) = {minutes}")
    return int(math.ceil(min_hold_hours * 60.0 / minutes))


def _utc_dates(index: pd.Index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return np.asarray(idx.date)


def _funding_ok_aligned(
    index: pd.Index,
    funding: pd.DataFrame | None,
    min_annual: float,
) -> np.ndarray:
    """True на барі t, якщо rolling річний funding (лише ≤ t) ≥ порогу.

    Немає історії ставок → усе False (fail closed).
    """
    n = len(index)
    if funding is None or funding.empty or "fundingRate" not in funding.columns:
        return np.zeros(n, dtype=bool)
    rates = funding["fundingRate"].astype(float).sort_index()
    window = _FUNDING_LOOKBACK_DAYS * 3
    annual = rates * _FUNDING_PERIODS_PER_YEAR
    rolled = annual.rolling(window, min_periods=max(3, _FUNDING_LOOKBACK_DAYS)).mean()
    aligned = rolled.reindex(index, method="ffill")
    ok = aligned.to_numpy(dtype=float, copy=False)
    return np.isfinite(ok) & (ok >= min_annual)


def apply_cell_overlay(
    signals: pd.Series,
    policy: CellPolicy,
    *,
    interval: str,
    funding: pd.DataFrame | None = None,
) -> pd.Series:
    """Повернути сигнали після enable / short-clip / funding-sleep / min_hold / throttle."""
    if not policy.enabled:
        return pd.Series(0.0, index=signals.index, dtype=float)

    raw = signals.astype(float).reindex(signals.index).fillna(0.0).clip(-1.0, 1.0)
    if not policy.allow_short:
        raw = raw.clip(lower=0.0)

    if policy.enabled_when == "funding_regime":
        fund_ok = _funding_ok_aligned(signals.index, funding, policy.funding_annual_min)
    else:
        fund_ok = np.ones(len(signals), dtype=bool)

    min_hold = hours_to_bars(policy.min_hold_hours, interval)
    max_tpd = policy.max_trades_per_day
    dates = _utc_dates(signals.index) if max_tpd is not None else None
    desired = raw.to_numpy(dtype=float, copy=False)

    pos = 0.0
    held = 0
    trades_today = 0
    day_token: object | None = None
    result = np.zeros(len(desired), dtype=float)

    for i, want in enumerate(desired):
        if dates is not None:
            d = dates[i]
            if d != day_token:
                day_token = d
                trades_today = 0

        target = 0.0 if not fund_ok[i] else float(want)

        if pos != 0.0 and held < min_hold:
            if target == 0.0:
                pos = 0.0
                held = 0
            elif (target > 0) != (pos > 0):
                held += 1
            else:
                pos = target
                held += 1
            result[i] = pos
            continue

        is_entry = pos == 0.0 and target != 0.0
        is_flip = pos != 0.0 and target != 0.0 and (pos > 0) != (target > 0)
        if (is_entry or is_flip) and max_tpd is not None and trades_today >= max_tpd:
            result[i] = pos
            if pos != 0.0:
                held += 1
            continue

        if is_entry or is_flip:
            trades_today += 1
            pos = target
            held = 1
        elif target == 0.0:
            pos = 0.0
            held = 0
        else:
            pos = target
            held += 1
        result[i] = pos

    return pd.Series(result, index=signals.index, dtype=float)
