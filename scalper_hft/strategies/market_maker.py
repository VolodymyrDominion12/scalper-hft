"""Пасивний market maker (експериментальний): post-only котирування обох сторін.

Теорія (книга, гл. 15 — Noncontractual Market Making):
    - прибуток = спред; витрати = maker-комісія; головний ризик = adverse selection;
    - retail-перевага лише за низьких maker-комісій (Binance 0.02%, rebates на VIP).

Реалізація (спрощена, для бектесту на свічках):
    - коливання ціни → філл лімітного ордера моделюється ймовірнісно:
      якщо бар перетинає рівень котирування, ордер заповнюється з adverse
      selection херкатом (множник 0.5 — гірша ціна заповнення);
    - інвентар обмежений: якщо позиція > inventory_cap — не котируємо далі
      у цьому напрямку; існуючу позицію закриваємо наступним кросом.

⚠ Експериментальний: реальна черга (queue position), спайки та ребейти
не моделюються. Для production — nautilus_trader зі справжнім L2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.strategies.base import Strategy


class PassiveMarketMaker(Strategy):
    name = "market_maker"
    family = "market_making"
    preferred_regimes = frozenset({"low", "normal"})
    requires = frozenset({"l2"})

    param_space = {
        "spread_offset_mult": (0.1, 1.5, 0.1),
        "inventory_cap": (0.0, 2.0, 0.25),
        "quote_size_pct": (0.005, 0.05, 0.005),
        "adverse_sel_haircut": (0.3, 1.0, 0.1),
        "gamma": (0.01, 1.0, 0.05),
        "kappa": (0.5, 5.0, 0.5),
        "vpin_threshold": (0.5, 0.95, 0.05),
    }

    def __init__(
        self,
        spread_offset_mult: float = 0.5,
        inventory_cap: float = 1.0,
        quote_size_pct: float = 0.02,
        adverse_sel_haircut: float = 0.5,
        gamma: float = 0.1,
        kappa: float = 1.5,
        use_vpin_shield: bool = False,
        vpin_threshold: float = 0.75,
        use_liquidation_shield: bool = True,
        liquidation_threshold: float = 5_000_000.0,
    ) -> None:
        super().__init__(
            spread_offset_mult=spread_offset_mult,
            inventory_cap=inventory_cap,
            quote_size_pct=quote_size_pct,
            adverse_sel_haircut=adverse_sel_haircut,
            gamma=float(gamma),
            kappa=float(kappa),
            use_vpin_shield=bool(use_vpin_shield),
            vpin_threshold=float(vpin_threshold),
            use_liquidation_shield=bool(use_liquidation_shield),
            liquidation_threshold=float(liquidation_threshold),
        )

    def reservation_price(
        self,
        mid_or_micro: float,
        inventory: float,
        vol: float,
        time_horizon: float = 1.0,
    ) -> float:
        """Ціна бронювання (Avellaneda-Stoikov):

        r(s, q) = s - q * gamma * sigma^2 * (T - t)
        де s — mid або micro-price, q — поточний інвентар, gamma — несприйняття ризику,
        sigma — миттєва волатильність.
        """
        gamma = float(self.get("gamma", 0.1))
        # q * gamma * vol^2 * horizon
        skew = inventory * gamma * (vol**2) * time_horizon
        return float(mid_or_micro - skew)

    def optimal_half_spread(
        self,
        vol: float,
        time_horizon: float = 1.0,
    ) -> float:
        """Оптимальний напівспред (Avellaneda-Stoikov):

        delta = 0.5 * gamma * sigma^2 * (T - t) + (1 / gamma) * ln(1 + gamma / kappa)
        """
        gamma = float(self.get("gamma", 0.1))
        kappa = float(self.get("kappa", 1.5))
        if gamma <= 0 or kappa <= 0:
            return 0.0
        part1 = 0.5 * gamma * (vol**2) * time_horizon
        part2 = (1.0 / gamma) * float(np.log1p(gamma / kappa))
        return float(part1 + part2)

    def compute_quotes(
        self,
        mid_or_micro: float,
        inventory: float,
        vol: float,
        vpin: float | None = None,
        base_spread: float | None = None,
        liquidation_cascade: float = 0.0,
    ) -> tuple[float, float, bool]:
        """Обчислює рівні котирування (bid, ask, active).

        Якщо увімкнено VPIN-щит і токсичність > vpin_threshold, котирування
        призупиняються або спред подвоюється для захисту від adverse selection.
        Також додано захист від каскаду ліквідацій.
        """
        use_vpin = bool(self.get("use_vpin_shield", False))
        vpin_thresh = float(self.get("vpin_threshold", 0.75))
        use_liq = bool(self.get("use_liquidation_shield", True))
        liq_thresh = float(self.get("liquidation_threshold", 5_000_000.0))
        inv_cap = float(self.get("inventory_cap", 1.0))

        # Захист від токсичного потоку (VPIN circuit breaker)
        if use_vpin and vpin is not None and vpin > vpin_thresh:
            return 0.0, float("inf"), False

        # Захист від каскаду ліквідацій (зупиняємо маркетмейкінг на сильних рухах)
        if use_liq and liquidation_cascade > liq_thresh:
            return 0.0, float("inf"), False

        r = self.reservation_price(mid_or_micro, inventory, vol)
        if base_spread is not None and base_spread > 0:
            delta = base_spread * float(self.get("spread_offset_mult", 0.5))
        else:
            delta = self.optimal_half_spread(vol)

        bid = r - delta
        ask = r + delta

        # Обмеження інвентарю
        if inventory >= inv_cap:
            bid = 0.0  # не купуємо далі
        if inventory <= -inv_cap:
            ask = float("inf")  # не продаємо далі

        return float(bid), float(ask), True

    def generate_signals(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Повертає бажану позицію 0 завжди — market maker керується
        інвентарем, а не напрямком. Сигнали для нього генерує event_engine."""
        return pd.Series(0, index=df.index, dtype=int)
