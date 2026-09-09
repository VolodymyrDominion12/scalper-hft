"""Портфельне позиціонування: ERC + vol-targeting для live (Phase 2D).

Regime-scaled sizing через risk/portfolio layer: ERC-ваги (рівний внесок у
ризик) × vol-target scale (цільова портфельна волатильність) прив'язані до
supervisor ваг. Замість фіксованого position_pct — розмір залежить від
впевненості (regime) та волатильності.

Використовується live/pairs_runner при enable_vol_target=True.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def erc_vol_target_sizes(
    returns: pd.DataFrame,
    *,
    target_vol: float = 0.01,
    max_weight: float = 0.5,
    ann_factor: float | None = None,
) -> pd.DataFrame:
    """ERC-ваги × vol-target scale на кожному барі (каузально, expanding).

    Args:
        returns: (T × N) дохідності N символів/стратегій.
        target_vol: цільова портфельна волатильність (per bar або річна).
        max_weight: обмеження концентрації ERC.
        ann_factor: якщо задано — target_vol трактується як річна, scale =
            target_vol / (σ_port · √ann_factor). None = per-bar.

    Returns:
        (T × N) DataFrame ваг (сума по рядку ≤ 1), каузальних (expanding window).
    """
    from scalper_hft.portfolio.erc import erc_weights

    n = returns.shape[1]
    out = np.zeros_like(returns.values, dtype=float)
    vals = returns.values
    min_window = max(n * 5, 60)
    for t in range(len(returns)):
        if t < min_window:
            out[t] = np.full(n, 1.0 / n)
            continue
        # expanding window — лише минулі дані (без lookahead)
        w = erc_weights(vals[: t + 1], max_weight=max_weight)
        port = vals[: t + 1] @ w
        vol = float(np.std(port, ddof=0))
        if vol <= 0 or not np.isfinite(vol):
            scale = 1.0
        elif ann_factor is not None:
            scale = float(target_vol / (vol * np.sqrt(max(ann_factor, 1e-12))))
        else:
            scale = float(target_vol / vol) if vol > 0 else 1.0
        # масштаб не повинен перевищувати 1 (без плеча понад cap)
        scale = float(np.clip(scale, 0.0, 1.0))
        out[t] = w * scale
    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def regime_scaled_size(
    base_size: float,
    regime_confidence: float,
    *,
    vol_scale: float = 1.0,
    min_scale: float = 0.0,
    max_scale: float = 1.5,
) -> float:
    """Regime-scaled розмір позиції: base × confidence × vol_scale (2D).

    Args:
        base_size: базовий розмір (напр. position_pct × equity / price).
        regime_confidence: впевненість режиму ∈ [0, 1] (max HMM posterior).
        vol_scale: зворотній до волатильності (target_vol/realized_vol).
        min_scale/max_scale: кліп множника.

    Returns:
        Скаленований розмір позиції.
    """
    mult = float(np.clip(regime_confidence * vol_scale, min_scale, max_scale))
    return float(base_size) * mult


__all__ = ["erc_vol_target_sizes", "regime_scaled_size"]
