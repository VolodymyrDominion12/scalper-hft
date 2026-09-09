"""Єдиний append-only журнал спроб (trials) — замість «магічної» 50.

Проблема (overfitting-audit SKILL / Bailey–López de Prado): DSR коригує
спостережуваний Sharpe на кількість спроб (n_trials). Раніше n_trials брали
з `estimate_n_trials(combos, backtests_per_combo=50)` — де 50 була магічною
константою, що не відображала реальну кількість перебраних варіантів у сесії.

Цей модуль — append-only JSONL-журнал: кожен бектест/аудит/оптимізація
дописує рядок. Потім `count_trials(...)` дає чесну кількість спроб для
конкретної комірки (або глобально), яка стає нижньою межею n_trials для DSR.

Використання:
    record_trial(path, strategy="mean_reversion", symbol="BTCUSDT",
                 purpose="audit_cell", n_trials=50, score=0.42)
    n = count_trials(path, strategy="mean_reversion", symbol="BTCUSDT")
    # n_trials для DSR = max(estimate_n_trials(combos, 50), n)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT = Path("docs/reports/trial_ledger.jsonl")


def default_path() -> Path:
    return _DEFAULT


def _enabled_path(path: object) -> Path | None:
    """None або порожній/'.'-подібний шлях → журнал вимкнено (no-op).

    Path("") дорівнює Path("."): спроба відкрити його як файл кидає
    IsADirectoryError (це валило overfit-аудит, коли TRIAL_LEDGER_PATH
    не задано). Тут такі значення трактуємо як «журнал не налаштовано».
    """
    if path is None:
        return None
    p = Path(path)
    if p == Path(".") or str(path).strip() == "":
        return None
    return p


def record_trial(
    path: Path | None,
    *,
    strategy: str,
    symbol: str,
    purpose: str,
    n_trials: int = 1,
    score: float | None = None,
    extra: dict | None = None,
) -> None:
    """Дописати один запис про спробу (append-only). None/порожній path → no-op."""
    path = _enabled_path(path)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "strategy": strategy,
        "symbol": symbol,
        "purpose": purpose,
        "n_trials": int(n_trials),
        "score": score,
    }
    if extra:
        row["extra"] = extra
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_trials(
    path: Path | None,
    *,
    strategy: str | None = None,
    symbol: str | None = None,
    purpose: str | None = None,
) -> int:
    """Кількість записаних спроб (сума n_trials) за фільтром. None path → 0."""
    path = _enabled_path(path)
    if path is None or not path.exists():
        return 0
    total = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if strategy is not None and row.get("strategy") != strategy:
                continue
            if symbol is not None and row.get("symbol") != symbol:
                continue
            if purpose is not None and row.get("purpose") != purpose:
                continue
            total += int(row.get("n_trials", 1))
    return total


def effective_n_trials(
    ledger_path: Path | None,
    *,
    param_combinations: int,
    backtests_per_combo: int,
    strategy: str | None = None,
    symbol: str | None = None,
) -> int:
    """Чесна n_trials для DSR: max(оцінка з combos, реальний лічильник журналу).

    Якщо журнал порожній/відсутній — повертає оцінку з combos × backtests_per_combo
    (зворотна сумісність із «магічною» 50). Якщо журнал має більше — бере його
    (дослідник реально перебрав більше, ніж підказує формула).
    """
    from scalper_hft.validation.deflated_sharpe import estimate_n_trials

    estimate = estimate_n_trials(param_combinations, backtests_per_combo)
    ledger = count_trials(ledger_path, strategy=strategy, symbol=symbol)
    return max(estimate, ledger)


__all__ = ["default_path", "record_trial", "count_trials", "effective_n_trials"]
