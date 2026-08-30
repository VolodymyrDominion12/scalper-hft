"""One-Way Trading & Geometric Exit Ladders (Eyal Gofer Ch. 7).

Конкурентний аналіз та алгоритми One-Way Trading:
    - Замість очікування єдиного "ідеального" піку ціни (який неможливо вгадати без lookahead),
      алгоритм розбиває вихід на каскад рівнів L_i = P_entry * (1 + side * 2^i * K).
    - Кожному рівню відповідає геометрична частка обсягу q_i.
    - Гарантує стабільний competitive ratio проти оптимального оракула (offline optimal).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class LadderRung:
    level_idx: int
    target_price: float
    qty_fraction: float  # частка від початкової позиції [0..1]
    filled: bool = False


class OneWayTradingLadder:
    """Драбина виходу з позиції за геометричною прогресією."""

    def __init__(
        self,
        entry_price: float,
        side: int,  # +1 (Long), -1 (Short)
        base_step_pct: float = 0.002,  # 0.2% базовий крок K
        num_levels: int = 4,
        geometric_factor: float = 1.5,
    ) -> None:
        self.entry_price = entry_price
        self.side = side
        self.base_step_pct = base_step_pct
        self.num_levels = max(num_levels, 1)
        self.geometric_factor = geometric_factor
        self.rungs: list[LadderRung] = self._build_ladders()

    def _build_ladders(self) -> list[LadderRung]:
        """Створення рівнів ціни та часток обсягу."""
        # Геометричний розподіл часток обсягу
        weights = [1.0 / (self.geometric_factor**i) for i in range(self.num_levels)]
        norm_weights = [w / sum(weights) for w in weights]

        rungs = []
        for i in range(self.num_levels):
            # Ціновий зсув: side * base_step * (geometric_factor ^ i)
            step_mult = self.geometric_factor**i
            price_offset = self.side * self.entry_price * (self.base_step_pct * step_mult)
            target = self.entry_price + price_offset
            rungs.append(LadderRung(level_idx=i, target_price=target, qty_fraction=norm_weights[i]))
        return rungs

    def update_price(self, current_price: float) -> float:
        """Перевіряє, які рівні були досягнуті поточною ціною.

        Returns:
            Частка позиції, що має бути закрита на поточному кроці (0.0 .. 1.0).
        """
        closed_fraction = 0.0
        for rung in self.rungs:
            if not rung.filled:
                if self.side == 1 and current_price >= rung.target_price:
                    rung.filled = True
                    closed_fraction += rung.qty_fraction
                elif self.side == -1 and current_price <= rung.target_price:
                    rung.filled = True
                    closed_fraction += rung.qty_fraction
        return closed_fraction

    @property
    def remaining_fraction(self) -> float:
        """Частка позиції, що залишається відкритою."""
        return sum(r.qty_fraction for r in self.rungs if not r.filled)

    @property
    def is_fully_closed(self) -> bool:
        """Чи закриті всі рівні драбини."""
        return all(r.filled for r in self.rungs)
