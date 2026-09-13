"""VIP-рівні комісій на Binance / Bybit USDT-M futures (дослідження §7.1).

Дослідження «Стратегії MFT Криптоторгівлі 2026» §7.1 наголошує: для MFT з 50
угод/день комісії визначають прибутковість. Binance VIP 4–9 дає зниження
maker аж до 0%; Bybit Supreme VIP — 0.000% maker. Також BNB-дисконт 25% на
споті (на ф'ючерсах — окрема програма).

Цей модуль — таблиця рівнів + функція `resolve_fees(tier, exchange)`,
що повертає (maker_fee, taker_fee) як частки ноціоналу. Використовується
`CostModel.from_settings` коли `FEE_TIER` задано (не `vip0`).

Джерела ставок (2026):
    Binance USDT-M futures: https://www.binance.com/en/fee/futuresFee
    Bybit derivatives:     https://www.bybit.com/en/help-center/article/Trading-Fee-Structure

⚠ Ставки можуть змінюватись — перед live перевіряти актуальні на сайті біржі.
"""

from __future__ import annotations

from collections.abc import Mapping

# (maker_fee, taker_fee) як частки ноціоналу (0.0002 = 0.02%).
# Дослідження §7.1: Binance базові 0.020% maker / 0.050% taker; VIP 9 → 0% maker.
BINANCE_USDTM_TIERS: dict[str, tuple[float, float]] = {
    "vip0": (0.00020, 0.00050),  # базовий (без BNB)
    "vip1": (0.00016, 0.00040),
    "vip2": (0.00014, 0.00035),
    "vip3": (0.00012, 0.00032),
    "vip4": (0.00010, 0.00030),
    "vip5": (0.00008, 0.00027),
    "vip6": (0.00006, 0.00024),
    "vip7": (0.00004, 0.00020),
    "vip8": (0.00002, 0.00018),
    "vip9": (0.00000, 0.00017),  # найвищий — 0% maker
}

# Bybit derivatives: базові 0.020% maker / 0.055% taker; Supreme VIP 0.000% maker.
BYBIT_DERIVATIVES_TIERS: dict[str, tuple[float, float]] = {
    "vip0": (0.00020, 0.00055),  # базовий
    "vip1": (0.00016, 0.00050),
    "vip2": (0.00014, 0.00045),
    "vip3": (0.00012, 0.00040),
    "vip4": (0.00010, 0.00035),
    "vip5": (0.00008, 0.00030),
    "supreme": (0.00000, 0.00030),  # Supreme VIP — 0.000% maker
}

# Мапа біржа → таблиця рівнів.
_TIER_TABLES: Mapping[str, dict[str, tuple[float, float]]] = {
    "binance": BINANCE_USDTM_TIERS,
    "binanceusdm": BINANCE_USDTM_TIERS,
    "bybit": BYBIT_DERIVATIVES_TIERS,
}

# Біржі, для яких відомі таблиці (для валідації FEE_TIER).
SUPPORTED_EXCHANGES = frozenset(_TIER_TABLES.keys())

# BNB-дисконт на Binance (25% від комісії) — опційно, через окремий прапорець.
BNB_DISCOUNT = 0.25


def resolve_fees(
    tier: str,
    exchange: str = "binanceusdm",
    *,
    bnb_discount: bool = False,
) -> tuple[float, float]:
    """Повертає (maker_fee, taker_fee) для рівня `tier` на біржі `exchange`.

    Args:
        tier: ім'я рівня (`vip0`..`vip9`, `supreme`). Регистронезалежний.
        exchange: ідентифікатор біржі (`binance`, `binanceusdm`, `bybit`).
        bnb_discount: застосувати BNB-дисконт 25% (лише Binance).

    Returns:
        (maker_fee, taker_fee) як частки ноціоналу.

    Raises:
        ValueError: якщо рівень або біржа невідомі.
    """
    ex_key = str(exchange).lower().strip()
    table = _TIER_TABLES.get(ex_key)
    if table is None:
        raise ValueError(f"Невідома біржа для FEE_TIER: '{exchange}'. Підтримувані: {sorted(SUPPORTED_EXCHANGES)}")
    tier_key = str(tier).lower().strip()
    fees = table.get(tier_key)
    if fees is None:
        raise ValueError(f"Невідомий FEE_TIER '{tier}' для біржі '{exchange}'. Доступні: {sorted(table.keys())}")
    maker, taker = fees
    if bnb_discount and ex_key in ("binance", "binanceusdm"):
        # BNB-дисконт знижує обидві комісії на 25% (Binance futures).
        maker = maker * (1.0 - BNB_DISCOUNT)
        taker = taker * (1.0 - BNB_DISCOUNT)
    return maker, taker


def is_supported_tier(tier: str, exchange: str = "binanceusdm") -> bool:
    """Чи існує рівень `tier` для біржі `exchange`."""
    ex_key = str(exchange).lower().strip()
    table = _TIER_TABLES.get(ex_key)
    if table is None:
        return False
    return str(tier).lower().strip() in table


def available_tiers(exchange: str = "binanceusdm") -> list[str]:
    """Список доступних рівнів для біржі (сортований)."""
    ex_key = str(exchange).lower().strip()
    table = _TIER_TABLES.get(ex_key)
    if table is None:
        return []
    return sorted(table.keys())


__all__ = [
    "BINANCE_USDTM_TIERS",
    "BYBIT_DERIVATIVES_TIERS",
    "BNB_DISCOUNT",
    "resolve_fees",
    "is_supported_tier",
    "available_tiers",
    "SUPPORTED_EXCHANGES",
]
