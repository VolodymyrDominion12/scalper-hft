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
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.random.uniform(100, 1000, len(close)),
        }
    )
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
    assert unique_labels.issubset(valid_labels), f"Невалідні label-и: {unique_labels - valid_labels}"


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


def test_apply_min_dwell_suppresses_flapping() -> None:
    """Гістерезис гасить фліпи A→B→A без підтвердження."""
    from scalper_hft.features.regimes import apply_min_dwell

    idx = pd.date_range("2025-01-01", periods=8, freq="1h")
    flappy = pd.Series(["A", "B", "A", "B", "A", "B", "A", "B"], index=idx, dtype=object)
    # min_dwell=1 — без змін (кожен фліп миттєвий)
    assert list(apply_min_dwell(flappy, 1)) == list(flappy)
    # min_dwell=2 — жоден B не тримається 2 бари підряд → лишаємось у A
    assert list(apply_min_dwell(flappy, 2).unique()) == ["A"]
    # підтверджений перехід: B,B після A → перемикаємось на 2-му барі B
    series = pd.Series(["A", "A", "B", "B", "B", "A"], index=idx[:6], dtype=object)
    out = apply_min_dwell(series, 2)
    assert list(out) == ["A", "A", "A", "B", "B", "B"]


def test_regime_detector_dwell_reduces_switches(synthetic_close: pd.Series) -> None:
    """min_dwell_bars у детекторі не збільшує кількість змін структури."""
    from scalper_hft.features.regime_detector import RegimeDetector

    det0 = RegimeDetector(n_hmm_states=3, hmm_fit_bars=200, min_dwell_bars=0)
    det4 = RegimeDetector(n_hmm_states=3, hmm_fit_bars=200, min_dwell_bars=4)
    s0 = det0.detect(synthetic_close)["structure"]
    s4 = det4.detect(synthetic_close)["structure"]
    changes0 = int((s0 != s0.shift(1)).sum())
    changes4 = int((s4 != s4.shift(1)).sum())
    assert changes4 <= changes0, f"dwell мав би зменшити зміни: {changes4} > {changes0}"


def test_regime_detector_htf_structure_valid_labels(synthetic_close: pd.Series) -> None:
    """htf_structure у детекторі дає валідні structure/label (без помилок)."""
    from scalper_hft.features.regime_detector import RegimeDetector
    from scalper_hft.features.regimes import STRUCTURE_LABELS, VOL_LABELS

    idx = pd.date_range("2023-01-01", periods=4000, freq="1h")
    t = np.arange(4000, dtype=float)
    close = pd.Series(100.0 * np.exp(0.0003 * t) + np.sin(t / 60.0) * 3.0, index=idx)

    det = RegimeDetector(n_hmm_states=3, hmm_fit_bars=200, htf_structure="1d")
    result = det.detect(close)
    assert set(result["structure"].unique()) <= STRUCTURE_LABELS
    assert set(result["label"].unique()) <= {f"{s}|{v}" for s in STRUCTURE_LABELS for v in VOL_LABELS}


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
        det._hmm.means_,
        loaded._hmm.means_,
        rtol=1e-6,  # type: ignore
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


@pytest.mark.parametrize("blend_mode", ["regime_soft", "contextual_hedge", "exp3", "best_prior", "risk_overlay"])
def test_regime_supervisor_signals_shape(blend_mode: str, synthetic_df: pd.DataFrame) -> None:
    """generate_signals() повертає Series правильної форми для всіх blend_mode."""
    from scalper_hft.strategies import get_strategy

    sup = get_strategy(
        "regime_supervisor",
        strategies="mean_reversion,supertrend",
        blend_mode=blend_mode,
        n_hmm_states=3,
        hmm_fit_bars=200,
        min_dwell_bars=2,
    )
    sigs = sup.generate_signals(synthetic_df)

    assert isinstance(sigs, pd.Series)
    assert len(sigs) == len(synthetic_df)
    assert sigs.index.equals(synthetic_df.index)
    assert sigs.isna().sum() == 0, "Сигнали не мають містити NaN"
    assert (sigs >= -1.0).all() and (sigs <= 1.0).all(), "Сигнали мають бути в [-1, 1]"
    if blend_mode == "best_prior":
        uniq = set(sigs.dropna().unique())
        assert uniq <= {-1.0, 0.0, 1.0}, "best_prior має повертати нативні сигнали {-1, 0, 1}"


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


# ──────────────────────────────────────────────────────────────────────────
# risk_overlay (v2.2, цикл RS iter15–iter16): режимний шар = масштаб експозиції
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def overlay_df() -> pd.DataFrame:
    """4h-ряд із чергуванням спокійних і вибухових ділянок (є режим vol=high)."""
    np.random.seed(7)
    n = 900
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    vol = np.where((np.arange(n) // 60) % 3 == 0, 1.6, 0.25)  # кожна 3-тя ділянка — high-vol
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * vol), index=idx)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.uniform(100, 1000, n),
        }
    )


def test_risk_overlay_uses_frozen_sleeve_pool() -> None:
    """risk_overlay без явних strategies бере ЗАМОРОЖЕНИЙ пул 5 рукавів (RS-2)."""
    from scalper_hft.strategies.regime_supervisor import RISK_OVERLAY_CHILDREN, RegimeSupervisor

    sup = RegimeSupervisor(blend_mode="risk_overlay", hmm_fit_bars=200)
    assert sup.sub_strategies == [
        "ts_momentum",
        "ts_momentum",
        "cross_momentum",
        "funding_carry",
        "supertrend",
    ]
    # а для селекторних режимів дефолтний пул НЕ змінюється
    sup_soft = RegimeSupervisor(blend_mode="regime_soft", hmm_fit_bars=200)
    assert "supertrend" in sup_soft.sub_strategies
    assert RISK_OVERLAY_CHILDREN  # константа існує і не порожня


def test_risk_overlay_scales_exposure_in_high_vol(overlay_df: pd.DataFrame) -> None:
    """risk-on → рівновага рукавів; risk-off (vol=high) → scale × incumbent."""
    from scalper_hft.strategies.regime_supervisor import DEFAULT_RISK_OFF_SCALE, RegimeSupervisor

    sup = RegimeSupervisor(blend_mode="risk_overlay", hmm_fit_bars=200, min_dwell_bars=0)
    sigs = sup.generate_signals(overlay_df)
    regime = sup._detector.detect(overlay_df["close"])
    sleeves = sup._collect_signals(overlay_df, None, None)
    incumbent = sleeves.iloc[:, 0]
    equal = sleeves.mean(axis=1)

    risk_off = (regime["vol"].reindex(sigs.index).fillna("normal") == "high")
    assert risk_off.any(), "синтетика має містити режим vol=high"
    assert (~risk_off).any()

    np.testing.assert_allclose(
        sigs[risk_off].to_numpy(),
        (DEFAULT_RISK_OFF_SCALE * incumbent[risk_off]).to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(sigs[~risk_off].to_numpy(), equal[~risk_off].to_numpy(), atol=1e-12)
    assert sigs.isna().sum() == 0
    assert (sigs.abs() <= 1.0).all()


def test_risk_overlay_custom_scale(overlay_df: pd.DataFrame) -> None:
    """risk_off_scale масштабує рівно лінійно (0.5 → удвічі більше за 0.25)."""
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor

    sup_low = RegimeSupervisor(blend_mode="risk_overlay", hmm_fit_bars=200, risk_off_scale=0.25)
    sup_high = RegimeSupervisor(blend_mode="risk_overlay", hmm_fit_bars=200, risk_off_scale=0.5)
    regime = sup_low._detector.detect(overlay_df["close"])
    risk_off = (regime["vol"].reindex(overlay_df.index).fillna("normal") == "high")
    low = sup_low.generate_signals(overlay_df)[risk_off]
    high = sup_high.generate_signals(overlay_df)[risk_off]
    np.testing.assert_allclose(high.to_numpy(), 2.0 * low.to_numpy(), atol=1e-12)


def test_risk_overlay_no_lookahead(overlay_df: pd.DataFrame) -> None:
    """Мутація МАЙБУТНІХ барів не змінює сигнали минулого (лаг на рішенні)."""
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor

    sup = RegimeSupervisor(blend_mode="risk_overlay", hmm_fit_bars=200)
    cutoff = 450
    base = sup.generate_signals(overlay_df)

    mutated = overlay_df.copy()
    mutated.loc[overlay_df.index[cutoff]:, ["close", "open", "high", "low"]] *= 1.35
    after = sup.generate_signals(mutated)

    np.testing.assert_allclose(
        base.iloc[:cutoff].to_numpy(),
        after.iloc[:cutoff].to_numpy(),
        atol=1e-12,
        err_msg="risk_overlay має бути каузальним: майбутні бари не впливають на минулі сигнали",
    )


def test_strategy_param_parser_handles_boolean_false() -> None:
    """Регресія v2.2: `name:allow_short=False` має ВИМКАТИ шорти, а не вмикати.

    До фіксу значення лишалось рядком, а `bool("False") is True` — тому
    `ts_momentum:allow_short=False` фактично торгував у шорт (саме це розходження
    виявив reproducibility-гейт RS-3: SR 0.77 замість 1.16 на тих самих даних).
    """
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor

    parse = RegimeSupervisor._parse_param_value
    assert parse("False") is False
    assert parse("false") is False
    assert parse("no") is False
    assert parse("True") is True
    assert parse("20") == 20
    assert parse("1.5") == 1.5
    assert parse("abc") == "abc"

    # Поведінково: перший рукав замороженого пулу не має шортів, другий має.
    np.random.seed(3)
    n = 700
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.6), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0}
    )
    sup = RegimeSupervisor(blend_mode="risk_overlay", hmm_fit_bars=200)
    sleeves = sup._collect_signals(df, None, None)
    assert (sleeves.iloc[:, 0] >= 0).all(), "incumbent-рукав (allow_short=False) не має шортів"
    assert (sleeves.iloc[:, 1] < 0).any(), "контрольний рукав (allow_short=True) має мати шорти"
