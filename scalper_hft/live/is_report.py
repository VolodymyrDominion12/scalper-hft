"""Implementation Shortfall агрегатор (Phase B2)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class ISReport:
    n_fills: int
    median_is_bps: float
    maker_is_bps: float
    taker_is_bps: float
    model_slippage_bps: float
    coverage_ok: bool   # ≥ 20 fills для калібровки

    def summary(self) -> str:
        delta = self.median_is_bps - self.model_slippage_bps
        arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "≈")
        return (
            f"IS Report: {self.n_fills} fills | "
            f"Realized={self.median_is_bps:.1f} bps {arrow} Model={self.model_slippage_bps:.1f} bps\n"
            f"  Maker={self.maker_is_bps:.1f} bps | Taker={self.taker_is_bps:.1f} bps"
        )
