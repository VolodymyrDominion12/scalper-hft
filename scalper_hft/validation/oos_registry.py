"""Реєстр «спаленого» OOS (Narang гл. 9 — burning data)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

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
        if not line.startswith("|") or line.startswith("|---") or "strategy" in line.lower():
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5:
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


def append_usage(
    path: Path,
    strategy: str,
    symbol: str,
    start: date,
    end: date,
    purpose: str,
) -> None:
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
