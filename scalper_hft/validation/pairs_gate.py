"""Пороги PASS для pairs_arb (не directional Sharpe 0.3).

Directional `cell_audit.OOS_SHARPE_MIN = 0.3` відсік би поточний pairs SRh.
Цей модуль — задокументований pairs-гейт. Запис PASS у verdict_store робить
дослідник після WF/CSCV; раннер лише читає свіжий pair-PASS.
"""

from __future__ import annotations

# Частка walk-forward вікон з додатним OOS (STATUS LINK/BTC ≈ 65%).
WF_POS_FRAC_MIN = 0.55
# Combinatorially Symmetric CV: PBO < 0.5 (iter6b LINK/BTC = 0.000).
PBO_MAX = 0.50
# Рідкі maker-входи: нижче 20 угод на вікні аудиту — мала вибірка.
MIN_TRADES = 20
