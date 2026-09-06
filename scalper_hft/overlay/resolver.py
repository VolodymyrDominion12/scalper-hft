"""Резолвер Cell Overlay: exact → interval+cluster → wildcard.

Більший score перемагає; при нічиїй виграє пізніше правило в YAML
(можна перевизначити загальне вужчим унизу файлу).
"""

from __future__ import annotations

import inspect
from typing import Any

from scalper_hft.overlay.policy import CellMatch, CellPolicy, OverlayBook


def bind_strategy_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Лишити лише параметри, які __init__ стратегії реально приймає."""
    sig = inspect.signature(cls)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    known = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in known}


def _score(
    match: CellMatch, strategy: str, symbol: str, interval: str, clusters: dict[str, frozenset[str]]
) -> int | None:
    score = 0
    if match.wildcard_strategy:
        score += 1
    elif match.strategy == strategy:
        score += 8
    else:
        return None

    if match.wildcard_interval:
        score += 1
    elif interval in match.intervals:
        score += 4
    else:
        return None

    in_named_cluster = False
    if match.clusters:
        for name in match.clusters:
            members = clusters.get(name)
            if members is None:
                return None
            if symbol in members:
                in_named_cluster = True
                break
        if not in_named_cluster and not match.symbols:
            return None

    if match.wildcard_symbol:
        score += 1
    elif symbol in match.symbols:
        score += 8
        if match.clusters:
            if not in_named_cluster:
                return None
            score += 2
    elif in_named_cluster:
        score += 4
    else:
        return None

    return score


def resolve_policy(book: OverlayBook, strategy: str, symbol: str, interval: str) -> CellPolicy:
    """Повернути політику для клітинки. Без збігу правил — defaults."""
    best_score = -1
    winner: CellPolicy | None = None
    for rule in book.rules:
        scored = _score(rule.match, strategy, symbol, interval, book.clusters)
        if scored is None:
            continue
        if scored >= best_score:
            best_score = scored
            winner = rule.policy
    return winner if winner is not None else book.defaults
