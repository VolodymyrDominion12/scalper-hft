"""Модель транзакційних витрат (книга, гл. 5).

Три компоненти:
    1. Комісії (maker/taker) — фіксовані частки ноціоналу;
    2. Slippage — погіршення ціни виконання (bps);
    3. Market impact — для retail-розмірів на Binance ф'ючерсах нехтовно малий
       (глибокий стакан), тому за замовчуванням 0, але параметризується.

Важливо: для скальпінгу комісії + slippage часто перевищують очікуваний
прибуток — це головний фільтр життєздатності стратегії (див. RESEARCH.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Вартість одного round-trip = (fee + slippage) × 2 (вхід і вихід)."""

    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage_frac: float = 0.0002  # 2 bps
    impact_frac: float = 0.0

    def taker_cost_per_side(self) -> float:
        return self.taker_fee + self.slippage_frac + self.impact_frac

    def maker_cost_per_side(self) -> float:
        return self.maker_fee + self.impact_frac  # лімітні ордери без slippage

    def round_trip_taker(self) -> float:
        return 2 * self.taker_cost_per_side()

    def round_trip_maker(self) -> float:
        return 2 * self.maker_cost_per_side()

    def cost(self, side_is_maker: bool) -> float:
        return self.maker_cost_per_side() if side_is_maker else self.taker_cost_per_side()

    @property
    def breakeven_move_pct(self) -> float:
        """Мінімальний рух ціни (%), що покриває round-trip taker — бар'єр для скальпера."""
        return self.round_trip_taker() * 100
