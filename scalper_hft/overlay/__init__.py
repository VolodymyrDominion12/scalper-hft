"""Cell Overlay: політика клітинки між Alpha і Risk."""

from __future__ import annotations

from scalper_hft.overlay.apply import apply_cell_overlay, hours_to_bars
from scalper_hft.overlay.loader import default_overlay_path, load_overlay_book, overlay_book_from_dict
from scalper_hft.overlay.policy import CellMatch, CellPolicy, CellRule, OverlayBook
from scalper_hft.overlay.resolver import bind_strategy_kwargs, resolve_policy

__all__ = [
    "CellMatch",
    "CellPolicy",
    "CellRule",
    "OverlayBook",
    "apply_cell_overlay",
    "bind_strategy_kwargs",
    "default_overlay_path",
    "hours_to_bars",
    "load_overlay_book",
    "overlay_book_from_dict",
    "resolve_policy",
]
