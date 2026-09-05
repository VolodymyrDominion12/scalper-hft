"""Валідатор YAML-специфікацій стратегій scalper-hft.

Перевіряє кожну specs/strategies/<name>.yaml на відповідність схемі
(_schema.yaml). Викликається з tests/test_strategy_specs.py.

Використання:
    from specs.strategies._validator import load_spec, validate_spec, SpecError

    spec = load_spec("specs/strategies/pairs_arb.yaml")
    errors = validate_spec(spec)
    assert not errors, errors
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import yaml

# ── Константи схеми ────────────────────────────────────────────────────────

VALID_STATUSES = {"validated", "candidate", "pending", "rejected"}

VALID_FAMILIES = {
    "momentum",
    "mean_reversion",
    "relative_value",
    "carry",
    "flow",
    "market_making",
    "ml",
    "meta",
}

VALID_REGIME_LABELS = {"range", "trend_up", "trend_down", "low", "normal", "high"}

VALID_SIGNAL_VALUES = frozenset({-1, 0, 1})

# Обов'язкові поля за статусом
REQUIRED_BY_STATUS: dict[str, list[str]] = {
    "validated": ["name", "version", "status", "family", "hypothesis", "edge_conditions", "params", "invariants"],
    "candidate": ["name", "version", "status", "family", "hypothesis", "params", "invariants"],
    "pending":   ["name", "version", "status", "family", "hypothesis"],
    "rejected":  ["name", "version", "status", "family", "rejection_reason"],
}

MIN_EDGE_CONDITIONS: dict[str, int] = {
    "validated": 3,
    "candidate": 2,
    "pending": 0,
    "rejected": 0,
}

SPECS_DIR = Path(__file__).parent


class SpecError(ValueError):
    """Помилка валідації специфікації."""


def load_spec(path: str | Path) -> dict[str, Any]:
    """Завантажує YAML-специфікацію з файлу."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SpecError(f"{path}: файл порожній або не є YAML-словником")
    return data


def validate_spec(spec: dict[str, Any], *, path: str = "<unknown>") -> list[str]:
    """Перевіряє специфікацію на відповідність схемі.

    Повертає список рядків з помилками (порожній = OK).
    """
    errors: list[str] = []
    ctx = path  # для читабельних повідомлень

    # ── 1. Обов'язкові базові поля ─────────────────────────────────────────
    for field in ("name", "version", "status", "family"):
        if field not in spec:
            errors.append(f"[{ctx}] Відсутнє обов'язкове поле: '{field}'")

    if errors:
        return errors  # без цих полів неможливо продовжити

    # ── 2. Перевірка enum-полів ────────────────────────────────────────────
    status = spec.get("status", "")
    if status not in VALID_STATUSES:
        errors.append(f"[{ctx}] status '{status}' не входить у {sorted(VALID_STATUSES)}")
        return errors

    family = spec.get("family", "")
    if family not in VALID_FAMILIES:
        errors.append(f"[{ctx}] family '{family}' не входить у {sorted(VALID_FAMILIES)}")

    # ── 3. Обов'язкові поля за статусом ──────────────────────────────────
    required = REQUIRED_BY_STATUS.get(status, [])
    for field in required:
        if field not in spec:
            errors.append(f"[{ctx}] status='{status}' вимагає поле '{field}'")

    # ── 4. Перевірка edge_conditions ─────────────────────────────────────
    min_ec = MIN_EDGE_CONDITIONS.get(status, 0)
    ec = spec.get("edge_conditions", [])
    if min_ec > 0:
        if not isinstance(ec, list):
            errors.append(f"[{ctx}] edge_conditions має бути списком, отримано {type(ec).__name__}")
        elif len(ec) < min_ec:
            errors.append(f"[{ctx}] status='{status}' вимагає >= {min_ec} edge_conditions, є {len(ec)}")
        else:
            for i, cond in enumerate(ec):
                if not isinstance(cond, dict):
                    errors.append(f"[{ctx}] edge_conditions[{i}]: має бути словником з 'name' і 'description'")
                    continue
                if "name" not in cond:
                    errors.append(f"[{ctx}] edge_conditions[{i}]: відсутнє поле 'name'")
                if "description" not in cond:
                    errors.append(f"[{ctx}] edge_conditions[{i}]: відсутнє поле 'description'")

    # ── 5. Перевірка params ───────────────────────────────────────────────
    params = spec.get("params")
    if params is not None:
        if not isinstance(params, dict):
            errors.append(f"[{ctx}] params має бути словником")
        else:
            for pname, pspec in params.items():
                if not isinstance(pspec, dict):
                    errors.append(f"[{ctx}] params.{pname}: має бути словником з 'default'")
                    continue
                if "default" not in pspec:
                    errors.append(f"[{ctx}] params.{pname}: відсутнє поле 'default'")

    # ── 6. Перевірка invariants ───────────────────────────────────────────
    invariants = spec.get("invariants")
    if invariants is not None:
        if not isinstance(invariants, dict):
            errors.append(f"[{ctx}] invariants має бути словником")
        else:
            if "no_lookahead" not in invariants:
                errors.append(f"[{ctx}] invariants.no_lookahead — обов'язкове поле")
            elif invariants["no_lookahead"] is not True:
                errors.append(f"[{ctx}] invariants.no_lookahead ПОВИНЕН бути true (будь-яка стратегія)")

            if "signal_values" not in invariants:
                errors.append(f"[{ctx}] invariants.signal_values — обов'язкове поле")
            else:
                sv = invariants["signal_values"]
                if not isinstance(sv, list) or set(sv) != VALID_SIGNAL_VALUES:
                    errors.append(f"[{ctx}] invariants.signal_values має бути [-1, 0, 1], отримано {sv}")

    # ── 7. Перевірка preferred_regimes ───────────────────────────────────
    pr = spec.get("preferred_regimes", [])
    if pr is not None and isinstance(pr, list):
        unknown = set(pr) - VALID_REGIME_LABELS
        if unknown:
            errors.append(f"[{ctx}] preferred_regimes містить невідомі мітки: {sorted(unknown)}")

    # ── 8. rejection_reason для rejected ─────────────────────────────────
    if status == "rejected" and not spec.get("rejection_reason"):
        errors.append(f"[{ctx}] status='rejected' вимагає непорожній rejection_reason")

    # ── 9. name відповідає назві файлу ───────────────────────────────────
    if path != "<unknown>":
        file_stem = Path(path).stem
        if not file_stem.startswith("_") and spec.get("name") != file_stem:
            errors.append(
                f"[{ctx}] spec.name='{spec.get('name')}' не збігається з назвою файлу '{file_stem}.yaml'"
            )

    return errors


def validate_all_specs(specs_dir: str | Path | None = None) -> dict[str, list[str]]:
    """Перевіряє всі *.yaml-специфікації у директорії (крім _*.yaml).

    Повертає {path: [errors]} (порожній список = OK).
    """
    specs_dir = Path(specs_dir) if specs_dir else SPECS_DIR
    results: dict[str, list[str]] = {}

    pattern = str(specs_dir / "*.yaml")
    for path in sorted(glob.glob(pattern)):
        if os.path.basename(path).startswith("_"):
            continue
        try:
            spec = load_spec(path)
            errs = validate_spec(spec, path=path)
        except Exception as exc:  # noqa: BLE001
            errs = [f"[{path}] Помилка парсингу YAML: {exc}"]
        results[path] = errs

    return results


if __name__ == "__main__":
    results = validate_all_specs()
    total_errors = 0
    for path, errs in results.items():
        if errs:
            print(f"\n❌ {path}:")
            for e in errs:
                print(f"   {e}")
            total_errors += len(errs)
        else:
            print(f"✅ {path}")
    print(f"\n{'Всі специфікації валідні.' if total_errors == 0 else f'Знайдено {total_errors} помилок.'}")
