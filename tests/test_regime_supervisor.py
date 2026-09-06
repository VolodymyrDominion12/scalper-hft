"""Тести RegimeDetector та RegimeSupervisor.

Перевіряємо:
1. RegimeDetector: без lookahead (fit тільки на prefix).
2. RegimeDetector.step() — онлайн інкрементальний режим.
3. ContextualHedgeBlend: ваги оновлюються per-regime.
4. RegimeSupervisor: всі три blend_mode генерують сигнали без lookahead.
5. RegimeSupervisor зареєстрований у REGISTRY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_close() -> pd.Series:
    """Синтетичний ряд цін (trend + range + trend)."""
    np.random.seed(42)
    n = 1000
    t = np.arange(n)
    prices = 100.0 + np.sin(t / 50) * 5 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.Series(prices, name="close")


@pytest.fixture()
def synthetic_df(synthetic_close: pd.Series) -> pd.DataFrame:
    """DataFrame у форматі бектесту."""
    close = synthetic_close
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": np.random.uniform(100, 1000, len(close)),
    })
    return df


# ──────────────────────────────────────────────────────────────────────────
# RegimeDetector tests
# ──────────────────────────────────────────────────────────────────────────


def test_regime_detector_detect_shape(synthetic_close: pd.Series) -> None:
    """detect() повертає DataFrame з правильними колонками та розміром."""
    from scalper_hft.features.regime_detector import RegimeDetector

    det = RegimeDetector(n_hmm_states=3, hmm_fit_bars=200)
    result = det.detect(synthetic_close)

    assert isinstance(result, pd.DataFrame)
    assert "structure" in result.columns
    assert "vol" in result.columns
    assert "label" in result.columns
    assert "hmm_state" in result.columns
    assert "confidence" in result.columns
    assert len(result) == len(synthetic_close)


def test_regime_detector_no_lookahead(synthetic_close: pd.Series) -> None:
    """HMM навчається лише на prefix, не на всьому ряді.

    Критерій: якщо подати тільки першу половину ряду та повний ряд — HMM
    параметри (means_) мають бути ідентичними (навчені на тих самих fit_bars).
    """
    from scalper_hft.features.regime_detector import RegimeDetector

    fit_bars = 200
    det_full = RegimeDetector(n_hmm_states=3, hmm_fit_bars=fit_bars)
    det_full.fit(synthetic_close)

    det_half = RegimeDetector(n_hmm_states=3, hmm_fit_bars=fit_bars)
    det_half.fit(synthetic_close.iloc[:500])

    # Обидві моделі навчені на перших fit_bars барах — means_ мають збігатись
    assert det_full.is_fitted
    assert det_half.is_fitted
    np.testing.assert_allclose(
        det_full._hmm.means_,  # type: ignore[union-attr]
        det_half._hmm.means_,  # type: ignore[union-attr]
        rtol=1e-6,
        err_msg="HMM параметри відрізняються — порушення no-lookahead!",
    )


def test_regime_detector_valid_labels(synthetic_close: pd.Series) -> None:
    """Усі label-и мають бути валідними комбінаціями structure|vol."""
    from scalper_hft.features.regime_detector import RegimeDetector
    from scalper_hft.features.regimes import STRUCTURE_LABELS, VOL_LABELS

    det = RegimeDetector(n_hmm_states=3, hmm_fit_bars=200)
    result = det.detect(synthetic_close)

    valid_labels = {f"{s}|{v}" for s in STRUCTURE_LABELS for v in VOL_LABELS}
    # Деякі рядки можуть мати ffill-значення, але всі мають бути у valid_labels
    unique_labels = set(result["label"].unique())
    assert unique_labels.issubset(valid_labels), (
        f"Невалідні label-и: {unique_labels - valid_labels}"
    )


def test_regime_detector_step_basic(synthetic_close: pd.Series) -> None:
    """step() повертає RegimeState без помилок."""
    from scalper_hft.features.regime_detector import RegimeDetector, RegimeState

    det = RegimeDetector(n_hmm_states=3, hmm_fit_bars=200)
    state = None
    for price in synthetic_close.values[:300]:
        state = det.step(float(price))

    assert isinstance(state, RegimeState)
    assert state.structure in ("range", "trend_up", "trend_down")
    assert state.vol in ("low", "normal", "high")
    assert state.label == f"{state.structure}|{state.vol}"


def test_regime_detector_step_fits_after_warmup(synthetic_close: pd.Series) -> None:
    """HMM автоматично навчається після hmm_fit_bars барів у step()."""
    from scalper_hft.features.regime_detector import RegimeDetector

    det = RegimeDetector(n_hmm_states=3, hmm_fit_bars=300)
    assert not det.is_fitted
    for price in synthetic_close.values[:305]:
        det.step(float(price))
    assert det.is_fitted, "HMM має бути навченим після hmm_fit_bars барів"


def test_regime_detector_save_load(tmp_path, synthetic_close: pd.Series) -> None:
    """Збереження та завантаження моделі."""
    from scalper_hft.features.regime_detector import RegimeDetector

    det = RegimeDetector(n_hmm_states=3, hmm_fit_bars=200)
    det.fit(synthetic_close)

    path = str(tmp_path / "detector.pkl")
    det.save(path)

    loaded = RegimeDetector.load(path)
    assert loaded.is_fitted
    np.testing.assert_allclose(
        det._hmm.means_, loaded._hmm.means_, rtol=1e-6  # type: ignore
    )


# ──────────────────────────────────────────────────────────────────────────
# ContextualHedgeBlend tests
# ──────────────────────────────────────────────────────────────────────────


def test_contextual_hedge_blend_basic() -> None:
    """ContextualHedgeBlend ініціалізується і blend() повертає скаляр."""
    from scalper_hft.strategies.blend import ContextualHedgeBlend

    blend = ContextualHedgeBlend(n_experts=3, regimes=["range", "trend_up", "trend_down"])
    signals = np.array([0.5, -0.3, 0.8])
    result = blend.blend(signals, regime="range")
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0  # не кліпнуто в blend(), але ваги нормовані


def test_contextual_hedge_per_regime_update() -> None:
    """Оновлення ваг відбувається тільки у відповідному режимі."""
    from scalper_hft.strategies.blend import ContextualHedgeBlend

    blend = ContextualHedgeBlend(n_experts=2, regimes=["range", "trend_up"])
    initial_trend_weights = blend.weights("trend_up").copy()

    # Оновлюємо тільки в "range" — ваги "trend_up" не мають змінитись
    for _ in range(20):
        blend.step(np.array([0.1, -0.1]), regime="range")

    np.testing.assert_allclose(
        initial_trend_weights,
        blend.weights("trend_up"),
        rtol=1e-6,
        err_msg="Ваги trend_up змінились від оновлень у режимі range!",
    )


def test_contextual_hedge_bar_counts() -> None:
    """bar_counts рахує правильно по режимах."""
    from scalper_hft.strategies.blend import ContextualHedgeBlend

    blend = ContextualHedgeBlend(n_experts=2, regimes=["range", "trend_up"])
    for _ in range(5):
        blend.step(np.array([0.1, 0.2]), regime="range")
    for _ in range(3):
        blend.step(np.array([0.3, -0.1]), regime="trend_up")

    assert blend.bar_counts["range"] == 5
    assert blend.bar_counts["trend_up"] == 3


def test_contextual_hedge_unknown_regime_fallback() -> None:
    """Невідомий режим використовує global fallback без помилки."""
    from scalper_hft.strategies.blend import ContextualHedgeBlend

    blend = ContextualHedgeBlend(n_experts=2, regimes=["range"])
    # "trend_down" не в списку — має використати fallback
    result = blend.blend(np.array([1.0, -1.0]), regime="trend_down")
    assert isinstance(result, float)


# ──────────────────────────────────────────────────────────────────────────
# RegimeSupervisor tests
# ──────────────────────────────────────────────────────────────────────────


def test_regime_supervisor_registry() -> None:
    """RegimeSupervisor зареєстрований у REGISTRY."""
    from scalper_hft.strategies import REGISTRY, get_strategy

    assert "regime_supervisor" in REGISTRY
    sup = get_strategy(
        "regime_supervisor",
        strategies="mean_reversion,supertrend",
        blend_mode="regime_soft",
    )
    assert sup.name == "regime_supervisor"


@pytest.mark.parametrize("blend_mode", ["regime_soft", "contextual_hedge", "exp3"])
def test_regime_supervisor_signals_shape(
    blend_mode: str, synthetic_df: pd.DataFrame
) -> None:
    """generate_signals() повертає Series правильної форми для всіх blend_mode."""
    from scalper_hft.strategies import get_strategy

    sup = get_strategy(
        "regime_supervisor",
        strategies="mean_reversion,supertrend",
        blend_mode=blend_mode,
        n_hmm_states=3,
        hmm_fit_bars=200,
    )
    sigs = sup.generate_signals(synthetic_df)

    assert isinstance(sigs, pd.Series)
    assert len(sigs) == len(synthetic_df)
    assert sigs.index.equals(synthetic_df.index)
    assert sigs.isna().sum() == 0, "Сигнали не мають містити NaN"
    assert (sigs >= -1.0).all() and (sigs <= 1.0).all(), "Сигнали мають бути в [-1, 1]"


def test_regime_supervisor_no_lookahead(synthetic_df: pd.DataFrame) -> None:
    """Сигнали supervisor-а на першій половині == сигнали на тій самій половині повного ряду.

    Якщо є lookahead — сигнали на першій половині зміняться при додаванні даних.
    """
    from scalper_hft.strategies import get_strategy

    half = len(synthetic_df) // 2
    df_half = synthetic_df.iloc[:half].copy()
    df_full = synthetic_df.copy()

    sup_half = get_strategy(
        "regime_supervisor",
        strategies="mean_reversion",
        blend_mode="regime_soft",
        n_hmm_states=3,
        hmm_fit_bars=100,
    )
    sup_full = get_strategy(
        "regime_supervisor",
        strategies="mean_reversion",
        blend_mode="regime_soft",
        n_hmm_states=3,
        hmm_fit_bars=100,
    )

    sigs_half = sup_half.generate_signals(df_half)
    sigs_full = sup_full.generate_signals(df_full).iloc[:half]

    # Сигнали на першій половині мають збігатись (HMM навчений на перших 100 барах)
    np.testing.assert_allclose(
        sigs_half.values,
        sigs_full.values,
        atol=1e-10,
        err_msg="Lookahead виявлено: сигнали на першій половині залежать від майбутніх даних!",
    )


def test_regime_supervisor_sub_strategies(synthetic_df: pd.DataFrame) -> None:
    """sub_strategies повертає список імен суб-стратегій."""
    from scalper_hft.strategies import get_strategy

    sup = get_strategy(
        "regime_supervisor",
        strategies="mean_reversion,supertrend",
    )
    assert set(sup.sub_strategies) == {"mean_reversion", "supertrend"}  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
# RegimeAnalysis tests
# ──────────────────────────────────────────────────────────────────────────


def test_supervisor_vs_baseline_basic(synthetic_close: pd.Series) -> None:
    """supervisor_vs_baseline повертає DataFrame з правильними стратегіями."""
    from scalper_hft.research.regime_analysis import supervisor_vs_baseline

    np.random.seed(0)
    baseline = {
        "strat_a": pd.Series(np.random.randn(len(synthetic_close)) * 0.001, index=synthetic_close.index),
        "strat_b": pd.Series(np.random.randn(len(synthetic_close)) * 0.001, index=synthetic_close.index),
    }
    supervisor = pd.Series(np.random.randn(len(synthetic_close)) * 0.001, index=synthetic_close.index)

    result = supervisor_vs_baseline(synthetic_close, baseline, supervisor)
    assert "strat_a" in result.index
    assert "strat_b" in result.index
    assert "regime_supervisor" in result.index
    assert "sharpe" in result.columns
    assert "profit_factor" in result.columns


def test_regime_transition_matrix(synthetic_close: pd.Series) -> None:
    """regime_transition_matrix повертає матрицю нормовану по рядках."""
    from scalper_hft.research.regime_analysis import regime_transition_matrix

    trans = regime_transition_matrix(synthetic_close, n_hmm_states=3, hmm_fit_bars=200)
    assert isinstance(trans, pd.DataFrame)
    # Рядки нормовані (сума ≈ 1)
    row_sums = trans.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, np.ones(len(row_sums)), atol=1e-9)
