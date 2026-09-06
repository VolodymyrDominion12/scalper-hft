"""Сторінки Streamlit-дашборду (st.navigation → app_pages/)."""

from __future__ import annotations

import importlib
import sys

_COMMON_ALIASES = (
    "scalper_hft.app_pages._common",
    "app_pages._common",
    "_common",
    "scalper_hft.app_pages._results",
    "app_pages._results",
    "_results",
)


def reload_shared() -> None:
    """Перечитати `_common` зі диску.

    Streamlit тримає модулі в `sys.modules` між rerun-ами. Після зміни
    `_common.py` сторінки імпортують нові імена зі старого об'єкта модуля
    (`JOB_KINDS`, `RESEARCH_SECTION`, …) і падають з ImportError.
    """
    for name in _COMMON_ALIASES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        importlib.reload(mod)
