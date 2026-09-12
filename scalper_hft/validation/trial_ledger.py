"""Єдиний append-only журнал спроб (trials) — замість «магічної» 50.

Проблема (overfitting-audit SKILL / Bailey–López de Prado): DSR коригує
спостережуваний Sharpe на кількість спроб (n_trials). Раніше n_trials брали
з `estimate_n_trials(combos, backtests_per_combo=50)`, де combos — ПОВНИЙ
декартів добуток `param_space`, а 50 — магічна константа: для supertrend це
давало ~2·10⁶ спроб на клітинку, яку взагалі не підбирали (DSR>0.95 вимагав
річного Sharpe ~4.7, тобто гейт був нездоланним за побудовою).

Цей модуль — append-only JSONL-журнал: кожен бектест/аудит/оптимізація
дописує рядок. Потім `count_trials(...)` дає чесну кількість спроб,
яка стає нижньою межею n_trials для DSR.

Використання:
    record_trial(path, strategy="mean_reversion", symbol="BTCUSDT",
                 purpose="audit_cell", n_trials=36, score=0.42)
    n = count_trials(path, strategy="mean_reversion")
    # n_trials для DSR = max(варіанти цього аудиту, n)
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
    backtests_per_combo: int = 1,
    strategy: str | None = None,
    symbol: str | None = None,
) -> int:
    """Чесна n_trials для DSR: max(варіантів цього аудиту, лічильник журналу).

    `param_combinations` — кількість конфігурацій, які РЕАЛЬНО прогнали в межах
    цього аудиту (baseline + сітка sensitivity + CSCV-варіанти), а не розмір
    теоретичного простору параметрів `param_space`.

    ⚠️ Історія: раніше сюди передавали ПОВНИЙ декартів добуток `param_space`
    (напр. supertrend → 40 824) і множили на «магічну» `backtests_per_combo=50`,
    тобто n_trials ≈ 2·10⁶ навіть коли жодного підбору параметрів не робилося
    (audit_cell завжди запускає дефолтну конфігурацію). DSR з такою
    множинністю вимагає річного Sharpe ≈ 3.8–4.9 на OOS — гейт ставав
    нездоланним для будь-якої реальної стратегії, а вердикт втрачав
    розрізнювальну здатність (FAIL у 100% клітинок, `results/audit_verdicts.jsonl`).
    Тепер множинність = те, що справді перебрали: варіанти цього аудиту +
    накопичений журнал спроб (усі символи стратегії — вибір символу теж part of
    the search). Поріг DSR (0.95) не змінюється, лише прибирається фіктивне
    завищення числа спроб.
    """
    estimate = max(int(param_combinations), 1) * max(int(backtests_per_combo), 1)
    ledger = count_trials(ledger_path, strategy=strategy, symbol=symbol)
    return max(estimate, ledger, 1)


__all__ = ["default_path", "record_trial", "count_trials", "effective_n_trials"]
