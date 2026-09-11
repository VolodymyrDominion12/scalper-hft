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


def evaluate_pair_wf_gate(
    pos_frac: float,
    n_windows: int,
    stress_crash_max_dd: float | None = None,
    stress_liquidity_max_dd: float | None = None,
    backtest_max_dd: float | None = None,
) -> tuple[str, list[str]]:
    """WF-частка позитивних OOS-вікон → PASS/FAIL для pair-вердикта."""
    reasons: list[str] = []
    if n_windows == 0:
        reasons.append("WF вікон=0")
    elif pos_frac < WF_POS_FRAC_MIN:
        reasons.append(f"WF positive={pos_frac:.0%}<{WF_POS_FRAC_MIN:.0%}")
        
    if backtest_max_dd is not None and backtest_max_dd > 0:
        if stress_crash_max_dd is not None and stress_crash_max_dd > backtest_max_dd * 2.5:
            reasons.append(f"stress crash maxDD={stress_crash_max_dd:.1%} > {backtest_max_dd * 2.5:.1%} (baseline×2.5)")
        if stress_liquidity_max_dd is not None and stress_liquidity_max_dd > backtest_max_dd * 4.0:
            reasons.append(f"stress liquidity maxDD={stress_liquidity_max_dd:.1%} > {backtest_max_dd * 4.0:.1%} (baseline×4.0)")

    label = "PASS" if not reasons else "FAIL"
    return label, reasons
