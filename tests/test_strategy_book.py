"""STRATEGY_BOOK покриває всі імена REGISTRY (фаза H8)."""

from __future__ import annotations

from scalper_hft.research.strategy_book import STRATEGY_BOOK
from scalper_hft.strategies import REGISTRY


def test_strategy_book_covers_registry() -> None:
    names = {rec.name for rec in STRATEGY_BOOK}
    missing = sorted(set(REGISTRY) - names)
    assert missing == [], f"немає в STRATEGY_BOOK: {missing}"
