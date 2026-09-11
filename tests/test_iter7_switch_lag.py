"""Регресійні тести на lookahead у regime-switch аналізі (iter7).

Контекст (аудит 2026-09-11, знахідка K1). У `experiments/iter7_regime_analysis.py`
селектор «режим → стратегія» застосовувався так, що мітка режиму бару t вибирала
дохідність ТОГО САМОГО бару t. Оскільки `ret[t]` — дохідність інтервалу [t, t+1),
а мітка `reg[t]` рахується з `close[t]` (ціна на кінець того ж інтервалу), обидві
величини відомі одночасно і мітка не може вибирати «свою» дохідність.

Заявлений у документі «lag-1» був `out.shift(1)` на СЕРІЇ ДОХІДНОСТЕЙ — це дає
`te[m(reg[t-1])][t-1]`, тобто мітка й дохідність усе одно з одного бару. Заміряно
на реальних артефактах: опубліковане «shift(1)» → Sharpe +2.58, «без лага» → +2.59,
коректний лаг вибору → +0.41.

Ці тести фіксують правильну семантику: лаг ставиться на ВИБІР.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "iter7_regime_analysis.py"

PPY = 8760.0
REGIMES = ("range", "trend_up", "trend_down")


def _load_module():
    spec = importlib.util.spec_from_file_location("iter7_regime_analysis_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _sharpe(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    sd = float(r.std(ddof=0))
    if len(r) < 2 or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(PPY))


# ── 1. Механіка: лаг саме на виборі ─────────────────────────────────────────


def test_apply_choice_map_lags_the_choice_not_the_returns(mod):
    """`sel[t]` бере дохідність бару t за міткою бару t-1."""
    idx = pd.date_range("2025-01-01", periods=6, freq="1h")
    returns = pd.DataFrame({"single:a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)
    regime = pd.Series(["range", "range", "trend_up", "trend_up", "range", "trend_up"], index=idx)

    out = mod.apply_choice_map(returns, regime, {"trend_up": "single:a"})

    # Мітка trend_up стоїть на барах 2, 3, 5 → вибір діє на барах 3, 4, (6 поза межами).
    assert out.iloc[0] == 0.0  # попередній бар — range
    assert out.iloc[1] == 0.0
    assert out.iloc[2] == 0.0  # мітка бару 2 ще не діяла
    assert out.iloc[3] == 4.0  # мітка trend_up на барі 2 → дохідність бару 3
    assert out.iloc[4] == 5.0  # мітка trend_up на барі 3 → дохідність бару 4
    assert out.iloc[5] == 0.0  # мітка бару 4 — range


def test_apply_choice_map_unknown_or_empty_map_is_flat(mod):
    idx = pd.date_range("2025-01-01", periods=3, freq="1h")
    returns = pd.DataFrame({"single:a": [1.0, 2.0, 3.0]}, index=idx)
    regime = pd.Series(["trend_up"] * 3, index=idx)

    assert mod.apply_choice_map(returns, regime, {}).abs().sum() == 0.0
    assert mod.apply_choice_map(returns, regime, {"trend_up": None}).abs().sum() == 0.0
    # Стратегія, якої немає у матриці, не має зламати і не має дати значень.
    assert mod.apply_choice_map(returns, regime, {"trend_up": "single:missing"}).abs().sum() == 0.0


# ── 2. Синтетика, на якій same-bar вибір дає величезний фальшивий Sharpe ─────


def _leaky_dataset(n: int = 10_000, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Дохідності, де мітка бару t повністю визначає переможця бару t.

    single:a заробляє +1bp у trend_up і −1bp у trend_down; single:b — навпаки;
    single:c — лише шум. Мітки iid. Тому:
      * вибір за міткою ТОГО САМОГО бару дає +1bp майже на кожному барі → Sharpe ≫ 5;
      * вибір із лагом (мітка t-1) дає середнє ≈ 0 → Sharpe ≈ 0.

    До базового зсуву додано малий шум, інакше всередині режиму серія
    константна (sd = 0) і `sharpe()` повертає 0.0 — тоді вибір per-regime
    взагалі не спрацьовує і синтетика перестає бути чутливою до витоку.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    labels = rng.choice(np.array(REGIMES), size=n, p=[1 / 3, 1 / 3, 1 / 3])
    up = labels == "trend_up"
    down = labels == "trend_down"
    noise = rng.normal(0.0, 1e-6, n)
    a = np.where(up, 1e-4, np.where(down, -1e-4, 0.0)) + noise
    b = -a
    c = rng.normal(0.0, 1e-6, n)
    returns = pd.DataFrame({"single:a": a, "single:b": b, "single:c": c}, index=idx)
    return returns, pd.Series(labels, index=idx)


def _same_bar_selection(returns: pd.DataFrame, regime: pd.Series, mapping: dict) -> pd.Series:
    """ЕТАЛОН БАГА: мітка бару t вибирає дохідність бару t (lookahead на 1 бар)."""
    out = pd.Series(0.0, index=returns.index)
    for key, strat in mapping.items():
        if strat is None:
            continue
        pos = np.flatnonzero((regime == key).to_numpy())
        if len(pos):
            out.iloc[pos] = returns[strat].to_numpy()[pos]
    return out


def test_same_bar_selection_is_a_huge_fake_edge(mod):
    """Фіксуємо, що витік реальний: same-bar вибір дає фальшивий Sharpe ≫ 5."""
    returns, regime = _leaky_dataset()
    mapping = {"trend_up": "single:a", "trend_down": "single:b", "range": None}

    leaky = _sharpe(_same_bar_selection(returns, regime, mapping))
    causal = _sharpe(mod.apply_choice_map(returns, regime, mapping))

    assert leaky > 20.0, "синтетика має бути чутливою до витоку"
    # Багнута реалізація тут дає ≈131, каузальна — ≈1.0.
    assert abs(causal) < 5.0, f"каузальний вибір не має ловити ту саму дохідність (SR={causal:.2f})"


def test_shift_on_returns_does_not_remove_the_leak(mod):
    """`out.shift(1)` на дохідностях — НЕ виправлення (саме це було в коді)."""
    returns, regime = _leaky_dataset()
    mapping = {"trend_up": "single:a", "trend_down": "single:b", "range": None}

    leaky = _same_bar_selection(returns, regime, mapping)
    shifted = leaky.shift(1).fillna(0.0)

    assert _sharpe(shifted) > 20.0, "shift(1) на дохідностях лишає lookahead"
    assert abs(_sharpe(mod.apply_choice_map(returns, regime, mapping))) < 5.0


# ── 3. End-to-end: switch_test і rolling_switch ─────────────────────────────


def test_switch_test_does_not_leak_same_bar_regime(mod):
    returns, regime = _leaky_dataset()
    pool = list(returns.columns)

    res = mod.switch_test("SYNTH", returns, regime, pool)

    # На цій синтетиці same-bar вибір дає Sharpe ≈ 133 (перевірено), тому
    # будь-яке значення вище 5 означає, що лаг зламано.
    assert abs(res["switch_sharpe"]) < 5.0, res
    # oracle теж застосовується каузально (інакше він не верхня межа).
    assert abs(res["oracle_switch_sharpe"]) < 5.0, res


def test_rolling_switch_is_causal(mod):
    returns, regime = _leaky_dataset()
    pool = list(returns.columns)

    rows, series = mod.rolling_switch(returns, regime, pool, n_folds=4)

    assert rows, "мають бути фолди"
    assert not series.empty
    # Багнута реалізація дає тут ≈130; каузальна — порядку одиниць.
    for row in rows:
        assert abs(row["empirical_sharpe"]) < 5.0, row
    assert abs(_sharpe(series["empirical"])) < 5.0


def test_rolling_switch_best_single_baseline_is_not_lagged(mod):
    """Базова лінія (одна стратегія) не має штучного зсуву на бар.

    `ret[t]` — це вже P&L позиції, вирішеної на t-1, тому `.shift(1)` на серії
    однієї стратегії зайвий. Раніше `best_single` зсувався, а `mean_all` — ні,
    тобто конкурента штучно послаблювали.
    """
    idx = pd.date_range("2024-01-01", periods=10_000, freq="1h")
    rng = np.random.default_rng(11)
    a = rng.normal(3e-4, 1e-3, len(idx))
    returns = pd.DataFrame({"single:a": a, "single:b": np.zeros(len(idx))}, index=idx)
    regime = pd.Series(rng.choice(np.array(REGIMES), size=len(idx)), index=idx)

    rows, series = mod.rolling_switch(returns, regime, ["single:a", "single:b"], n_folds=4)

    assert all(r["best_single_name"] == "single:a" for r in rows), rows
    test_index = series.index
    expected = returns["single:a"].reindex(test_index)
    assert np.allclose(series["best_single"].to_numpy(), expected.to_numpy()), (
        "best_single має бути нативною серією стратегії, без зсуву"
    )


# ── 4. PBO не має містити lookahead-селектор як кандидата ───────────────────


def test_dsr_table_primary_pbo_uses_singles_only(mod):
    from scalper_hft.validation.cscv import pbo_cscv

    returns, regime = _leaky_dataset()
    pool = list(returns.columns)
    oos_by_symbol = {"SYNTH": (returns, regime, pd.Series("normal", index=returns.index))}

    table = mod.dsr_table(oos_by_symbol, pool)
    assert len(table) == 1

    _rows, series = mod.rolling_switch(returns, regime, pool, n_folds=4)
    singles = returns[pool].reindex(series.index).fillna(0.0).to_numpy().T
    expected = float(pbo_cscv(singles, n_blocks=8, purge_bars=0).pbo)

    assert table.loc[0, "pbo_cscv"] == pytest.approx(expected)
    # Друга колонка існує і рахується окремо (селектор похідний від singles,
    # тому він не є незалежною спробою — але його видно для повноти).
    assert "pbo_with_selector" in table.columns
    assert np.isfinite(table.loc[0, "pbo_with_selector"])


def test_dsr_table_selector_is_not_an_independent_candidate(mod):
    """Якщо селектор не додавати, PBO не може «вигравати» через власний ряд.

    На синтетиці із iid-мітками всі стратегії мають Sharpe ≈ 0, тому PBO
    (частка сплітів, де IS-кращий нижче медіани OOS) має бути близьким до 0.5,
    а не до 0.0, як давав lookahead-ряд.
    """
    returns, regime = _leaky_dataset()
    pool = list(returns.columns)
    oos_by_symbol = {"SYNTH": (returns, regime, pd.Series("normal", index=returns.index))}

    table = mod.dsr_table(oos_by_symbol, pool)
    assert 0.2 <= table.loc[0, "pbo_cscv"] <= 0.8, table.to_dict("records")
