"""YAML-адаптер OverlayBook (інфраструктура, не домен)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from scalper_hft.overlay.policy import (
    CellMatch,
    CellPolicy,
    CellRule,
    EnabledWhen,
    ExecutionMode,
    OverlayBook,
)

DEFAULT_OVERLAY_RELATIVE = Path("configs/cells/default.yaml")


def default_overlay_path() -> Path:
    """Шлях до bundled playbook відносно кореня репозиторію."""
    return Path(__file__).resolve().parents[2] / DEFAULT_OVERLAY_RELATIVE


def _as_str_set(raw: object) -> tuple[frozenset[str], bool]:
    if raw is None or raw == "*":
        return frozenset(), True
    if isinstance(raw, str):
        return (frozenset(), True) if raw == "*" else (frozenset({raw}), False)
    if isinstance(raw, list):
        vals = [str(x) for x in raw]
        if "*" in vals:
            return frozenset(), True
        return frozenset(vals), False
    raise TypeError(f"очікував str | list | '*', отримав {type(raw).__name__}")


def _policy_from_mapping(raw: dict[str, Any], base: CellPolicy) -> CellPolicy:
    execution = str(raw.get("execution", base.execution))
    if execution not in ("maker", "taker"):
        raise ValueError(f"execution має бути maker|taker, отримано {execution!r}")
    enabled_when = str(raw.get("enabled_when", base.enabled_when))
    if enabled_when not in ("always", "funding_regime"):
        raise ValueError(f"enabled_when має бути always|funding_regime, отримано {enabled_when!r}")
    max_tpd_raw = raw.get("max_trades_per_day", base.max_trades_per_day)
    max_tpd: float | None
    if max_tpd_raw is None or max_tpd_raw == "":
        max_tpd = None
    else:
        max_tpd = float(max_tpd_raw)
        if max_tpd <= 0:
            raise ValueError("max_trades_per_day має бути > 0 або null")
    min_hold = float(raw.get("min_hold_hours", base.min_hold_hours))
    if min_hold < 0:
        raise ValueError("min_hold_hours не може бути від'ємним")
    size_mult = float(raw.get("size_mult", base.size_mult))
    if not 0.0 <= size_mult <= 1.0:
        raise ValueError("size_mult має бути в [0, 1]")
    funding_min = float(raw.get("funding_annual_min", base.funding_annual_min))
    return CellPolicy(
        enabled=bool(raw.get("enabled", base.enabled)),
        allow_short=bool(raw.get("allow_short", base.allow_short)),
        execution=cast(ExecutionMode, execution),
        min_hold_hours=min_hold,
        max_trades_per_day=max_tpd,
        cost_gate=bool(raw.get("cost_gate", base.cost_gate)),
        enabled_when=cast(EnabledWhen, enabled_when),
        funding_annual_min=funding_min,
        size_mult=size_mult,
    )


def _match_from_mapping(raw: dict[str, Any], cluster_names: set[str]) -> CellMatch:
    strat_raw = raw.get("strategy", "*")
    wildcard_strategy = strat_raw in (None, "*")
    strategy = "*" if wildcard_strategy else str(strat_raw)
    intervals, wildcard_interval = _as_str_set(raw.get("interval", "*"))
    symbols, wild_sym = _as_str_set(raw.get("symbol"))
    clusters, wild_cl = _as_str_set(raw.get("cluster"))
    unknown = clusters - cluster_names
    if unknown:
        raise ValueError(f"невідомі кластери в match: {sorted(unknown)}")
    wildcard_symbol = wild_sym and wild_cl
    return CellMatch(
        strategy=strategy,
        intervals=intervals,
        symbols=symbols,
        clusters=clusters,
        wildcard_strategy=wildcard_strategy,
        wildcard_interval=wildcard_interval,
        wildcard_symbol=wildcard_symbol,
    )


def overlay_book_from_dict(raw: dict[str, Any]) -> OverlayBook:
    clusters_raw = raw.get("clusters") or {}
    if not isinstance(clusters_raw, dict):
        raise TypeError("clusters має бути mapping")
    clusters = {str(k): frozenset(str(s) for s in v) for k, v in clusters_raw.items()}
    defaults = _policy_from_mapping(raw.get("defaults") or {}, CellPolicy())
    rules: list[CellRule] = []
    for i, cell in enumerate(raw.get("cells") or []):
        if not isinstance(cell, dict):
            raise TypeError(f"cells[{i}] має бути mapping")
        match_raw = cell.get("match") or {}
        if not isinstance(match_raw, dict):
            raise TypeError(f"cells[{i}].match має бути mapping")
        match = _match_from_mapping(match_raw, set(clusters))
        policy = _policy_from_mapping(cell, defaults)
        rules.append(CellRule(match=match, policy=policy))
    return OverlayBook(clusters=clusters, defaults=defaults, rules=tuple(rules))


def load_overlay_book(path: str | Path | None = None) -> OverlayBook:
    if path is None:
        p = default_overlay_path()
    else:
        p = Path(path)
        bundled = default_overlay_path()
        if not p.exists() and bundled.exists() and p.name == bundled.name:
            p = bundled
    if not p.exists():
        raise FileNotFoundError(f"overlay YAML не знайдено: {p}")
    loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError("корінь overlay YAML має бути mapping")
    return overlay_book_from_dict(loaded)
