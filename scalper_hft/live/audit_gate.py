"""Hard-гейт overfitting-аудиту для paper/live (overfitting-audit skill).

Два пороги (не змішувати):
  - Directional (`cell_audit`): avg OOS Sharpe > 0.3, DSR, smoothness, PBO.
  - Pairs (`pairs_arb` 1h maker): комірка `LEG1/LEG2` у verdict_store;
    метрики — частка WF-вікон > 0, PBO, n_trades (див. pairs_gate.py).
    SRh ≈ 0.006 НЕ порівнюється з OOS_SHARPE_MIN.

PASS-вердикт більше не advisory: перед стартом `paper-run-pairs` / live
раннер перевіряє свіжий PASS для пари. Fail-closed: немає вердикта, FAIL
або протухлий → старт заборонено.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from scalper_hft.validation.verdict_store import latest_pair_verdict, latest_verdict, pair_cell

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
    """Перевірка гейту для однієї directional-комірки. Повертає (ok, повідомлення)."""
    v = latest_verdict(strategy, symbol, interval, path=path)
    return _verdict_ok(v, f"{strategy} {symbol} {interval}", max_age_days=max_age_days, now=now)


def audit_gate_check_pair(
    strategy: str,
    leg1: str,
    leg2: str,
    interval: str,
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    path: Path | None = None,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Перевірка гейту для пари (не двох ніг окремо)."""
    cell = pair_cell(leg1, leg2)
    v = latest_pair_verdict(strategy, cell, interval, path=path)
    return _verdict_ok(v, f"{strategy} {cell} {interval}", max_age_days=max_age_days, now=now)


def _verdict_ok(
    v: dict | None,
    label: str,
    *,
    max_age_days: int,
    now: datetime | None,
) -> tuple[bool, str]:
    if v is None:
        return False, (
            f"немає записаного аудиту для {label} — запустіть `uv run python -m scalper_hft.cli overfit` спочатку"
        )
    status = str(v.get("label") or "")
    if status != "PASS":
        return False, f"вердикт {status}: {v.get('reasons') or '—'}"
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
    """Fail-closed для directional live: raise, якщо хоч один символ без PASS."""
    failures: list[str] = []
    for symbol in symbols:
        ok, msg = audit_gate_check(strategy, symbol, interval, max_age_days=max_age_days, path=path)
        if not ok:
            failures.append(f"{symbol}: {msg}")
    if failures:
        raise RuntimeError("Overfitting-гейт FAIL (live заборонено до PASS):\n  " + "\n  ".join(failures))


def require_pair_audit_pass(
    strategy: str,
    leg1: str,
    leg2: str,
    interval: str,
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    path: Path | None = None,
) -> None:
    """Fail-closed для pairs paper/live: комірка пари, не дві ноги окремо."""
    ok, msg = audit_gate_check_pair(strategy, leg1, leg2, interval, max_age_days=max_age_days, path=path)
    if not ok:
        raise RuntimeError(f"Overfitting-гейт пари FAIL ({leg1}/{leg2}): {msg}")
