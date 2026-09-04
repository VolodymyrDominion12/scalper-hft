"""Таксономія альфа-моделей (Narang гл. 3) і м'які ваги за режимом.

`family` і `preferred_regimes` на стратегії — гіпотези для дослідження,
не live-гейт. Порожній `preferred_regimes` означає «усі режими» (вага 1.0).
Клас підтверджується лише умовною OOS-метрикою, не інтуїцією.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

from scalper_hft.features.regimes import STRUCTURE_LABELS, VOL_LABELS

AlphaFamily = Literal[
    "momentum",
    "mean_reversion",
    "relative_value",
    "carry",
    "flow",
    "market_making",
    "ml",
    "meta",
]

PreferredRegime = Literal["range", "trend_up", "trend_down", "low", "normal", "high"]

FAMILIES: Final[frozenset[str]] = frozenset(get_args(AlphaFamily))
PREFERRED_REGIME_LABELS: Final[frozenset[str]] = STRUCTURE_LABELS | VOL_LABELS
DEFAULT_UNFAVORABLE_WEIGHT: Final[float] = 0.25


def regime_capital_weight(
    preferred: frozenset[str],
    structure: str,
    vol: str,
    *,
    unfavorable: float = DEFAULT_UNFAVORABLE_WEIGHT,
) -> float:
    """М'яка вага капіталу в [unfavorable, 1.0].

    Теги структури (`range` / `trend_up` / `trend_down`) і волатильності
    (`low` / `normal` / `high`) незалежні: невідповідність множить вагу на
    `unfavorable` (не обнуляє). Порожній preferred → завжди 1.0.
    """
    if not 0.0 <= unfavorable <= 1.0:
        raise ValueError(f"unfavorable must be in [0, 1], got {unfavorable}")
    unknown = preferred - PREFERRED_REGIME_LABELS
    if unknown:
        raise ValueError(f"unknown preferred regime labels: {sorted(unknown)}")

    weight = 1.0
    struct_tags = preferred & STRUCTURE_LABELS
    vol_tags = preferred & VOL_LABELS
    if struct_tags and structure not in struct_tags:
        weight *= unfavorable
    if vol_tags and vol not in vol_tags:
        weight *= unfavorable
    return weight
