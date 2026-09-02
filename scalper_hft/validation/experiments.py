"""Каталог дослідницьких експериментів: val → OOS лише після IMPROVED.

Дисципліна win_ai (одна зміна на експеримент, OOS не чіпаємо, доки val
не IMPROVED і є мінімум угод). Спалені OOS-вікна лишаються в oos_registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

from scalper_hft.validation.oos_registry import OosWindow, append_usage, is_burned

_DEFAULT_CATALOG = Path("docs/reports/experiments.md")
MIN_TRADES_DEFAULT = 20


class Verdict(StrEnum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class Experiment:
    id: str
    strategy: str
    hypothesis: str
    val_start: date
    val_end: date
    test_start: date
    test_end: date
    baseline_id: str | None = None


@dataclass(frozen=True, slots=True)
class ZoneMetrics:
    trades: int
    sharpe: float
    total_return: float
    max_drawdown: float = 0.0


def compare_to_baseline(
    candidate: ZoneMetrics,
    baseline: ZoneMetrics,
    *,
    min_sharpe_delta: float = 0.0,
) -> Verdict:
    """IMPROVED лише якщо Sharpe і return не гірші, а Sharpe строго більший на дельту."""
    better_sharpe = candidate.sharpe > baseline.sharpe + min_sharpe_delta
    not_worse_return = candidate.total_return >= baseline.total_return
    if better_sharpe and not_worse_return:
        return Verdict.IMPROVED
    if candidate.sharpe < baseline.sharpe or candidate.total_return < baseline.total_return:
        return Verdict.REGRESSED
    return Verdict.SKIPPED


def qualifies_for_oos(
    verdict: Verdict,
    trades: int,
    *,
    min_trades: int = MIN_TRADES_DEFAULT,
) -> bool:
    """OOS лише після IMPROVED на val і достатньої кількості угод."""
    return verdict is Verdict.IMPROVED and trades >= min_trades


def is_test_window_burned(experiment: Experiment, windows: list[OosWindow]) -> bool:
    """True, якщо test-вікно вже «спалене» для цієї стратегії."""
    symbol = experiment.id
    return is_burned(windows, experiment.strategy, symbol, experiment.test_start, experiment.test_end)


def record_oos_usage(experiment: Experiment, path: Path | None = None) -> None:
    """Позначити test-вікно спаленим після реального OOS-прогону."""
    append_usage(
        path or Path("docs/reports/oos_usage.md"),
        experiment.strategy,
        experiment.id,
        experiment.test_start,
        experiment.test_end,
        f"experiment:{experiment.id}",
    )


def parse_catalog(text: str) -> list[Experiment]:
    """Markdown-таблиця: id | strategy | hypothesis | val_start | val_end | test_start | test_end | baseline."""
    rows: list[Experiment] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7 or parts[0].lower() == "id":
            continue
        try:
            baseline = parts[7] if len(parts) > 7 and parts[7] not in {"", "-"} else None
            rows.append(
                Experiment(
                    id=parts[0],
                    strategy=parts[1],
                    hypothesis=parts[2],
                    val_start=date.fromisoformat(parts[3]),
                    val_end=date.fromisoformat(parts[4]),
                    test_start=date.fromisoformat(parts[5]),
                    test_end=date.fromisoformat(parts[6]),
                    baseline_id=baseline,
                )
            )
        except ValueError:
            continue
    return rows


def default_catalog_path() -> Path:
    return _DEFAULT_CATALOG


def load_catalog(path: Path | None = None) -> list[Experiment]:
    p = path or default_catalog_path()
    if not p.exists():
        return []
    return parse_catalog(p.read_text(encoding="utf-8"))
