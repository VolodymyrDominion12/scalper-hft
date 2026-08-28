"""Реєстр стратегій + функції завантаження за ім'ям."""

from __future__ import annotations

from scalper_hft.strategies.base import Strategy
from scalper_hft.strategies.basis_reversion import BasisReversion
from scalper_hft.strategies.cvd_momentum import CvdMomentumScalper
from scalper_hft.strategies.funding_arb import FundingArb
from scalper_hft.strategies.funding_carry import FundingCarryScalper
from scalper_hft.strategies.market_maker import PassiveMarketMaker
from scalper_hft.strategies.mean_reversion import MeanReversionScalper
from scalper_hft.strategies.ml_strategy import MLStrategy
from scalper_hft.strategies.ob_imbalance import ObImbalanceScalper
from scalper_hft.strategies.pairs_arb import PairsArb

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (
        MeanReversionScalper,
        CvdMomentumScalper,
        ObImbalanceScalper,
        PassiveMarketMaker,
        FundingCarryScalper,
        FundingArb,
        BasisReversion,
        PairsArb,
        MLStrategy,
    )
}


def get_strategy(name: str, **params: float | int | str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(
            f"Невідома стратегія '{name}'. Доступні: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**params)


__all__ = ["Strategy", "REGISTRY", "get_strategy"]
