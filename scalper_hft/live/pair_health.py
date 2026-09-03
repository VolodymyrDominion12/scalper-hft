"""Rolling ADF / OU half-life gate для paper pairs (без lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.validation.coint_scan import _adf_pvalue, _half_life

DEFAULT_ADF_MAX_P = 0.05
DEFAULT_MAX_HALF_LIFE = 200.0
DEFAULT_MIN_OBS = 80


def pair_entry_allowed(
    spread: pd.Series,
    *,
    adf_max_p: float = DEFAULT_ADF_MAX_P,
    max_half_life: float = DEFAULT_MAX_HALF_LIFE,
    min_obs: int = DEFAULT_MIN_OBS,
) -> tuple[bool, str]:
    """Чи можна відкривати нову позицію на закритому барі t.

    False якщо мало спостережень, ADF p > порогу або half-life inf / занадто довгий.
    Виходи (flatten) цим гейтом не блокуються — лише нові входи.
    """
    s = spread.dropna()
    if len(s) < min_obs:
        return True, "ok:мало_спостережень"
    p = float(_adf_pvalue(s))
    hl = float(_half_life(s))
    if not np.isfinite(p) or p > adf_max_p:
        return False, f"adf_kill p={p:.3f}"
    if not np.isfinite(hl) or hl <= 1.0 or hl > max_half_life:
        return False, f"hl_kill hl={hl}"
    return True, "ok"
