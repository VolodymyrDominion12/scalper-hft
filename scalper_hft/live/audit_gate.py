"""Hard-гейт overfitting-аудиту для paper/live (overfitting-audit skill).

PASS-вердикт `audit_cell` більше не advisory: перед стартом торгівлі раннер
перевіряє, що комірка (стратегія × символ × ТФ) має СВІЖИЙ записаний PASS
у `results/audit_verdicts.jsonl`. Fail-closed: немає вердикта, FAIL або
протухлий вердикт → старт заборонено (live) / гучне попередження (paper,
якщо REQUIRE_AUDIT_PASS=false).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from scalper_hft.validation.verdict_store import latest_verdict

DEFAULT_MAX_AGE_DAYS = 30


def audit_gate_check(
    strategy: str,
    symbol: str,
    interval: str,
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    path: Path | None = None,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Перевірка гейту для однієї комірки. Повертає (ok, повідомлення)."""
    v = latest_verdict(strategy, symbol, interval, path=path)
    if v is None:
        return False, (
            f"немає записаного аудиту для {strategy} {symbol} {interval} — "
            "запустіть `uv run python -m scalper_hft.cli overfit` спочатку"
        )
    label = str(v.get("label") or "")
    if label != "PASS":
        return False, f"вердикт {label}: {v.get('reasons') or '—'}"
    try:
        ts = datetime.fromisoformat(str(v["ts"]))
    except (KeyError, ValueError):
        return False, "битий timestamp вердикта"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age = (now or datetime.now(UTC)) - ts
    if age > timedelta(days=max_age_days):
        return False, f"вердикт протух: {age.days} діб > {max_age_days} — переаудітуйте комірку"
    return True, f"PASS від {ts.date()} ({age.days} дн. тому)"


def require_audit_pass(
    strategy: str,
    symbols: list[str],
    interval: str,
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    path: Path | None = None,
) -> None:
    """Fail-closed перевірка для live: raise RuntimeError, якщо хоч одна
    комірка (стратегія × символ) не має свіжого PASS."""
    failures: list[str] = []
    for symbol in symbols:
        ok, msg = audit_gate_check(strategy, symbol, interval, max_age_days=max_age_days, path=path)
        if not ok:
            failures.append(f"{symbol}: {msg}")
    if failures:
        raise RuntimeError("Overfitting-гейт FAIL (live заборонено до PASS):\n  " + "\n  ".join(failures))
