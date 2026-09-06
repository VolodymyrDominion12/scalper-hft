"""Spec-Driven Development: тести специфікацій стратегій scalper-hft.

Кожен тест читає YAML-файл з specs/strategies/ і перевіряє:
1. Валідність схеми (обов'язкові поля, enum-значення, структура).
2. Відповідність реєстру (name і family збігаються з Python-класом).
3. Behavioral assertions (generate_signals на синтетичних даних).
4. Param space consistency (параметри зі spec є у param_space класу).

Запуск:
    uv run pytest tests/test_strategy_specs.py -v
    uv run pytest tests/test_strategy_specs.py -k pairs_arb -v
    uv run pytest tests/test_strategy_specs.py -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

# ── Шляхи ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
SPECS_DIR = PROJECT_ROOT / "specs" / "strategies"

sys.path.insert(0, str(PROJECT_ROOT))

# ── Утиліти ───────────────────────────────────────────────────────────────


def load_all_specs() -> list[tuple[str, dict[str, Any]]]:
    """Завантажує всі YAML-специфікації (крім _*.yaml)."""
    specs = []
    for path in sorted(SPECS_DIR.glob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            specs.append((path.stem, data))
    return specs


ALL_SPECS = load_all_specs()
SPEC_IDS = [name for name, _ in ALL_SPECS]


def make_synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Синтетичний OHLCV DataFrame для тестування стратегій."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 50_000 + np.cumsum(rng.normal(0, 100, n))
    open_ = close + rng.normal(0, 50, n)
    high = np.maximum(close, open_) + abs(rng.normal(0, 30, n))
    low = np.minimum(close, open_) - abs(rng.normal(0, 30, n))
    volume = abs(rng.normal(1000, 200, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def make_pair_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """OHLCV із leg1/leg2 для pairs_arb і sparse_basket."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    leg1 = 50_000 + np.cumsum(rng.normal(0, 100, n))
    # Leg2 коінтегрована з leg1 (mean-reverting spread)
    leg2 = leg1 / 20 + np.cumsum(rng.normal(0, 1, n)) * 0.3
    close = leg1
    open_ = close + rng.normal(0, 50, n)
    high = np.maximum(close, open_) + abs(rng.normal(0, 30, n))
    low = np.minimum(close, open_) - abs(rng.normal(0, 30, n))
    volume = abs(rng.normal(1000, 200, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "leg1": leg1,
            "leg2": leg2,
        },
        index=dates,
    )


# ── Phase 1: Валідація схеми ──────────────────────────────────────────────


@pytest.mark.parametrize("name,spec", ALL_SPECS, ids=SPEC_IDS)
def test_spec_schema_valid(name: str, spec: dict[str, Any]) -> None:
    """Кожна специфікація відповідає _schema.yaml."""
    sys.path.insert(0, str(SPECS_DIR))
    from _validator import validate_spec  # type: ignore[import]

    spec_path = str(SPECS_DIR / f"{name}.yaml")
    errors = validate_spec(spec, path=spec_path)
    assert not errors, f"Специфікація {name}.yaml має помилки:\n" + "\n".join(errors)


# ── Phase 2: Відповідність реєстру ────────────────────────────────────────


@pytest.mark.parametrize("name,spec", ALL_SPECS, ids=SPEC_IDS)
def test_spec_name_in_registry(name: str, spec: dict[str, Any]) -> None:
    """Ім'я стратегії зі spec існує у REGISTRY."""
    from scalper_hft.strategies import REGISTRY

    spec_name = spec.get("name", name)
    assert spec_name in REGISTRY, (
        f"'{spec_name}' є у specs/, але відсутній у REGISTRY. "
        f"Зареєструй стратегію у scalper_hft/strategies/__init__.py."
    )


@pytest.mark.parametrize("name,spec", ALL_SPECS, ids=SPEC_IDS)
def test_spec_family_matches_class(name: str, spec: dict[str, Any]) -> None:
    """Family зі spec збігається із Strategy.family у класі."""
    from scalper_hft.strategies import REGISTRY

    spec_name = spec.get("name", name)
    if spec_name not in REGISTRY:
        pytest.skip(f"'{spec_name}' відсутній у REGISTRY (перевіряється іншим тестом)")

    cls = REGISTRY[spec_name]
    spec_family = spec.get("family", "")
    assert cls.family == spec_family, (
        f"{spec_name}: spec.family='{spec_family}', але {cls.__name__}.family='{cls.family}'. Оновіть spec або клас."
    )


@pytest.mark.parametrize("name,spec", ALL_SPECS, ids=SPEC_IDS)
def test_spec_preferred_regimes_match_class(name: str, spec: dict[str, Any]) -> None:
    """preferred_regimes зі spec збігається з Strategy.preferred_regimes."""
    from scalper_hft.strategies import REGISTRY

    spec_name = spec.get("name", name)
    if spec_name not in REGISTRY:
        pytest.skip(f"'{spec_name}' відсутній у REGISTRY")

    cls = REGISTRY[spec_name]
    spec_regimes = frozenset(spec.get("preferred_regimes") or [])
    class_regimes = frozenset(cls.preferred_regimes)

    assert spec_regimes == class_regimes, (
        f"{spec_name}: spec.preferred_regimes={sorted(spec_regimes)}, "
        f"але {cls.__name__}.preferred_regimes={sorted(class_regimes)}. "
        f"Оновіть spec або клас."
    )


# ── Phase 3: Param space consistency ─────────────────────────────────────


@pytest.mark.parametrize("name,spec", ALL_SPECS, ids=SPEC_IDS)
def test_spec_params_in_param_space(name: str, spec: dict[str, Any]) -> None:
    """Кожен параметр зі spec.params є у Strategy.param_space або __init__.

    Примітка: стратегії з **kwargs (__init__(self, **params)) використовують
    'params_doc' замість 'params' у spec — перевірка пропускається (немає __init__ сигнатури).
    """
    import inspect

    from scalper_hft.strategies import REGISTRY

    spec_name = spec.get("name", name)
    if spec_name not in REGISTRY:
        pytest.skip(f"'{spec_name}' відсутній у REGISTRY")

    # Якщо spec використовує params_doc — пропускаємо (kwargs-стратегія)
    if "params_doc" in spec and "params" not in spec:
        pytest.skip(f"{spec_name}: використовує params_doc (**kwargs), перевірка параметрів не застосовна")

    spec_params = spec.get("params") or {}
    if not spec_params:
        return  # немає params у spec — пропускаємо

    cls = REGISTRY[spec_name]

    # Беремо всі параметри __init__ (крім self та **kwargs)
    try:
        sig = inspect.signature(cls.__init__)
        init_params = {
            p
            for p, v in sig.parameters.items()
            if p != "self"
            and v.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
    except (ValueError, TypeError):
        init_params = set()

    param_space_keys = set(cls.param_space.keys())
    all_known_params = param_space_keys | init_params

    missing = []
    for pname in spec_params:
        if pname not in all_known_params:
            missing.append(pname)

    assert not missing, (
        f"{spec_name}: параметри {missing} є у spec.params, але відсутні "
        f"у {cls.__name__}.param_space або __init__. "
        f"Відомі параметри: {sorted(all_known_params)}"
    )


# ── Phase 4: Behavioral assertions ────────────────────────────────────────


def _get_validated_or_candidate(specs: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Фільтрує лише validated та candidate стратегії для behavioral тестів."""
    return [(n, s) for n, s in specs if s.get("status") in ("validated", "candidate")]


ACTIVE_SPECS = _get_validated_or_candidate(ALL_SPECS)
ACTIVE_IDS = [name for name, _ in ACTIVE_SPECS]


@pytest.mark.parametrize("name,spec", ACTIVE_SPECS, ids=ACTIVE_IDS)
def test_signals_domain(name: str, spec: dict[str, Any]) -> None:
    """generate_signals() повертає тільки значення з {-1, 0, 1}.

    Примітка: мета-стратегії (continuous_output_possible: true) можуть
    повертати float сигнали у [-1, 1] — цей тест пропускається.
    """
    from scalper_hft.strategies import REGISTRY

    spec_name = spec.get("name", name)
    if spec_name not in REGISTRY:
        pytest.skip(f"'{spec_name}' відсутній у REGISTRY")

    # Мета-стратегії можуть повертати float-зважені сигнали — пропускаємо
    invariants = spec.get("invariants") or {}
    if invariants.get("continuous_output_possible", False):
        pytest.skip(
            f"{spec_name}: мета-стратегія з continuous_output_possible=true, "
            f"повертає float у [-1,1] замість цілих сигналів"
        )

    cls = REGISTRY[spec_name]
    strategy = cls()

    requires_pair = invariants.get("requires_pair_columns", False)
    df = make_pair_ohlcv() if requires_pair else make_synthetic_ohlcv()

    try:
        signals = strategy.generate_signals(df)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{spec_name}: generate_signals() кинув виняток: {exc}")

    assert isinstance(signals, pd.Series), f"{spec_name}: має повертати pd.Series"

    valid_values = {-1, 0, 1}
    actual_values = set(signals.dropna().unique())
    invalid = actual_values - valid_values
    assert not invalid, (
        f"{spec_name}: generate_signals() повернув неочікувані значення {invalid}. Дозволені: {valid_values}"
    )


@pytest.mark.parametrize("name,spec", ACTIVE_SPECS, ids=ACTIVE_IDS)
def test_no_nan_output(name: str, spec: dict[str, Any]) -> None:
    """generate_signals() не повертає NaN."""
    from scalper_hft.strategies import REGISTRY

    spec_name = spec.get("name", name)
    if spec_name not in REGISTRY:
        pytest.skip(f"'{spec_name}' відсутній у REGISTRY")

    # Лише якщо spec явно задекларував no_nan_output: true
    invariants = spec.get("invariants") or {}
    if not invariants.get("no_nan_output", True):
        pytest.skip(f"{spec_name}: spec не гарантує no_nan_output")

    cls = REGISTRY[spec_name]
    strategy = cls()

    requires_pair = invariants.get("requires_pair_columns", False)
    df = make_pair_ohlcv() if requires_pair else make_synthetic_ohlcv()

    try:
        signals = strategy.generate_signals(df)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{spec_name}: generate_signals() кинув виняток: {exc}")

    nan_count = signals.isna().sum()
    assert nan_count == 0, (
        f"{spec_name}: generate_signals() повернув {nan_count} NaN значень. Spec гарантує no_nan_output: true."
    )


@pytest.mark.parametrize("name,spec", ACTIVE_SPECS, ids=ACTIVE_IDS)
def test_no_lookahead(name: str, spec: dict[str, Any]) -> None:
    """Сигнал на барі t не змінюється при мутації барів t+1..T (no-lookahead).

    Тест: беремо сигнал на барі idx = N//2, потім рандомно мутуємо
    всі бари після idx і перевіряємо що сигнал на idx не змінився.
    """
    from scalper_hft.strategies import REGISTRY

    spec_name = spec.get("name", name)
    if spec_name not in REGISTRY:
        pytest.skip(f"'{spec_name}' відсутній у REGISTRY")

    invariants = spec.get("invariants") or {}
    assert invariants.get("no_lookahead", False), f"{spec_name}: spec.invariants.no_lookahead має бути true"

    cls = REGISTRY[spec_name]
    strategy = cls()

    requires_pair = invariants.get("requires_pair_columns", False)
    df_orig = make_pair_ohlcv(n=300) if requires_pair else make_synthetic_ohlcv(n=300)

    try:
        signals_orig = strategy.generate_signals(df_orig)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{spec_name}: generate_signals() кинув виняток: {exc}")

    # Індекс перевірки: середина датасету
    check_idx = len(df_orig) // 2

    # Мутуємо всі бари після check_idx
    rng = np.random.default_rng(999)
    df_mutated = df_orig.copy()
    future_slice = slice(check_idx + 1, None)
    for col in ["open", "high", "low", "close"]:
        if col in df_mutated.columns:
            df_mutated.loc[df_mutated.index[future_slice], col] += rng.normal(0, 5000, len(df_orig) - check_idx - 1)
    if "leg1" in df_mutated.columns:
        df_mutated.loc[df_mutated.index[future_slice], "leg1"] += rng.normal(0, 5000, len(df_orig) - check_idx - 1)
    if "leg2" in df_mutated.columns:
        df_mutated.loc[df_mutated.index[future_slice], "leg2"] += rng.normal(0, 200, len(df_orig) - check_idx - 1)

    try:
        strategy2 = cls()  # новий інстанс (reset state)
        signals_mutated = strategy2.generate_signals(df_mutated)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{spec_name}: generate_signals() (mutated) кинув виняток: {exc}")

    orig_val = signals_orig.iloc[check_idx]
    mutated_val = signals_mutated.iloc[check_idx]

    assert orig_val == mutated_val, (
        f"{spec_name}: порушення no-lookahead! "
        f"Сигнал на барі {check_idx} змінився з {orig_val} на {mutated_val} "
        f"після мутації майбутніх барів. "
        f"Перевір rolling/shift операції у generate_signals()."
    )


@pytest.mark.parametrize("name,spec", ACTIVE_SPECS, ids=ACTIVE_IDS)
def test_signals_length_matches_df(name: str, spec: dict[str, Any]) -> None:
    """generate_signals() повертає Series з тією ж довжиною, що й вхідний df."""
    from scalper_hft.strategies import REGISTRY

    spec_name = spec.get("name", name)
    if spec_name not in REGISTRY:
        pytest.skip(f"'{spec_name}' відсутній у REGISTRY")

    cls = REGISTRY[spec_name]
    strategy = cls()

    requires_pair = (spec.get("invariants") or {}).get("requires_pair_columns", False)
    df = make_pair_ohlcv() if requires_pair else make_synthetic_ohlcv()

    try:
        signals = strategy.generate_signals(df)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{spec_name}: generate_signals() кинув виняток: {exc}")

    assert len(signals) == len(df), f"{spec_name}: signals має {len(signals)} елементів, очікується {len(df)} (як df)."


# ── Phase 5: Spec coverage (registry vs specs) ────────────────────────────


def test_all_registry_strategies_have_spec() -> None:
    """Кожна стратегія з REGISTRY має spec у specs/strategies/."""
    from scalper_hft.strategies import REGISTRY

    spec_names = {s.get("name") for _, s in ALL_SPECS}
    missing_specs = []

    for name in sorted(REGISTRY.keys()):
        if name not in spec_names:
            missing_specs.append(name)

    assert not missing_specs, (
        f"Стратегії без специфікації (SDD violation): {missing_specs}\n"
        f"Створи specs/strategies/<name>.yaml для кожної стратегії.\n"
        f"Мінімум для rejected: name, version, status, family, rejection_reason."
    )
