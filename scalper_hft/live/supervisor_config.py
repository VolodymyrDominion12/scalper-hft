"""Конфігурація RegimeSupervisor з YAML.

Книга, гл. 9 (Implementation): "Configuration Management is crucial for HFT".
Замість хардкоду словників у коді, стратегії мають описуватись декларативно.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class FamilyConfig:
    name: str
    allocation: float = 1.0


@dataclass
class StrategyConfig:
    id: str
    family: str
    preferred_regimes: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    allocation_weight: float = 1.0


@dataclass
class SupervisorConfig:
    name: str = "default_supervisor"
    families: list[FamilyConfig] = field(default_factory=list)
    strategies: list[StrategyConfig] = field(default_factory=list)
    max_total_leverage: float = 1.0
    risk_regime_filtering: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> SupervisorConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Конфіг {p} не знайдено.")

        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Parse families
        families = []
        for f_dict in raw.get("families", []):
            families.append(
                FamilyConfig(
                    name=f_dict.get("name", "unknown"),
                    allocation=float(f_dict.get("allocation", 1.0)),
                )
            )

        # Parse strategies
        strategies = []
        for s_dict in raw.get("strategies", []):
            strategies.append(
                StrategyConfig(
                    id=s_dict.get("id", "unknown"),
                    family=s_dict.get("family", "unknown"),
                    preferred_regimes=s_dict.get("preferred_regimes", []),
                    params=s_dict.get("params", {}),
                    allocation_weight=float(s_dict.get("allocation_weight", 1.0)),
                )
            )

        return cls(
            name=raw.get("name", "default_supervisor"),
            families=families,
            strategies=strategies,
            max_total_leverage=float(raw.get("max_total_leverage", 1.0)),
            risk_regime_filtering=bool(raw.get("risk_regime_filtering", True)),
        )
