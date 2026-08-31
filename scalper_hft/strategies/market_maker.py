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

import pandas as pd

from scalper_hft.strategies.base import Strategy


class PassiveMarketMaker(Strategy):
    name = "market_maker"

    param_space = {
        "spread_offset_mult": (0.1, 1.5, 0.1),
        "inventory_cap": (0.0, 2.0, 0.25),
        "quote_size_pct": (0.005, 0.05, 0.005),
        "adverse_sel_haircut": (0.3, 1.0, 0.1),
    }

    def __init__(
        self,
        spread_offset_mult: float = 0.5,
        inventory_cap: float = 1.0,
        quote_size_pct: float = 0.02,
        adverse_sel_haircut: float = 0.5,
    ) -> None:
        super().__init__(
            spread_offset_mult=spread_offset_mult,
            inventory_cap=inventory_cap,
            quote_size_pct=quote_size_pct,
            adverse_sel_haircut=adverse_sel_haircut,
        )

    def generate_signals(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Повертає бажану позицію 0 завжди — market maker керується
        інвентарем, а не напрямком. Сигнали для нього генерує event_engine."""
        return pd.Series(0, index=df.index, dtype=int)

