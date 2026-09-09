"""Реєстр «спаленого» OOS (Narang гл. 9 — burning data)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

_DEFAULT = Path("docs/reports/oos_usage.md")


@dataclass(frozen=True)
class OosWindow:
    strategy: str
    symbol: str
    start: date
    end: date
    purpose: str


def parse_registry(text: str) -> list[OosWindow]:
    rows: list[OosWindow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5 or parts[0].lower() == "strategy":
            continue
        try:
            rows.append(
                OosWindow(
                    parts[0],
                    parts[1],
                    date.fromisoformat(parts[2]),
                    date.fromisoformat(parts[3]),
                    parts[4],
                )
            )
        except ValueError:
            continue
    return rows


def is_burned(windows: list[OosWindow], strategy: str, symbol: str, start: date, end: date) -> bool:
    for w in windows:
        if w.strategy != strategy or w.symbol != symbol:
            continue
        if start <= w.end and end >= w.start:
            return True
    return False


def _enabled_path(path: Path | str | None) -> Path | None:
    """None або порожній/'.'-подібний шлях → реєстр вимкнено (no-op).

    Path("") == Path("."): відкривати його як файл не можна (IsADirectoryError).
    Виникає, коли OOS_REGISTRY_PATH задано порожнім рядком у env.
    """
    if path is None:
        return None
    p = Path(path)
    if p == Path(".") or str(path).strip() == "":
        return None
    return p


def append_usage(
    path: Path | str | None,
    strategy: str,
    symbol: str,
    start: date,
    end: date,
    purpose: str,
) -> None:
    path = _enabled_path(path)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Спалений OOS\n\n| strategy | symbol | start | end | purpose |\n|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    existing = parse_registry(path.read_text(encoding="utf-8"))
    if is_burned(existing, strategy, symbol, start, end):
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"| {strategy} | {symbol} | {start.isoformat()} | {end.isoformat()} | {purpose} |\n")


def default_path() -> Path:
    return _DEFAULT


def oos_range_from_df(df: pd.DataFrame, days: int) -> tuple[date, date] | None:
    """Діапазон дат OOS-вікна для датасету `df` (останні `days` днів).

    Повертає (start, end) календарні дати першого та останнього бару датасету.
    None, якщо df порожній або без DatetimeIndex. Використовується audit_cell для
    реєстрації «спаленого» OOS — увесь завантажений відрізок вважається OOS-зоною
    (дослідник завантажує саме той шматок, на якому приймає рішення).
    """
    if df is None or len(df) == 0:
        return None
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        # спробуємо перетворити
        try:
            idx = pd.to_datetime(idx)
        except Exception:
            return None
    start = pd.Timestamp(idx[0]).date()
    end = pd.Timestamp(idx[-1]).date()
    return start, end


def check_and_burn(
    *,
    strategy: str,
    symbol: str,
    df: object,
    days: int,
    purpose: str,
    registry_path: Path | None = None,
    enforce: bool = True,
) -> tuple[bool, str]:
    """Перевірити OOS-вікно на «спаленість» і (опційно) дописати використання.

    Повертає (ok, reason). Якщо enforce=True і вікно вже спалене — (False,
    reason). Якщо ok — дописує використання у реєстр (ідемпотентно) і повертає
    (True, ""). Якщо enforce=False — лише дописує без перевірки (завжди ok).

    Призначення: єдина точка OOS-дисципліни для audit_cell/overfit/sweep/optimize.
    """
    path = _enabled_path(registry_path if registry_path is not None else default_path())
    if path is None:
        # реєстр вимкнено (порожній/'.' шлях) — не блокуємо і не пишемо
        return True, ""
    rng = oos_range_from_df(df, days)  # type: ignore[arg-type]
    if rng is None:
        # без дат немає чого реєструвати — не блокуємо
        return True, ""
    start, end = rng
    existing = parse_registry(path.read_text(encoding="utf-8")) if path.exists() else []
    if enforce and is_burned(existing, strategy, symbol, start, end):
        return False, f"OOS-вікно вже спалене ({start}..{end}) для {strategy}/{symbol}"
    append_usage(path, strategy, symbol, start, end, purpose)
    return True, ""


__all__ = [
    "OosWindow",
    "parse_registry",
    "is_burned",
    "append_usage",
    "default_path",
    "oos_range_from_df",
    "check_and_burn",
]
