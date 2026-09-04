from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.features.regimes import (
    DEFAULT_TREND_THRESHOLD,
    STRUCTURE_LABELS,
    VOL_LABELS,
    named_market_state,
)
from scalper_hft.strategies.base import Strategy
from scalper_hft.strategies.taxonomy import DEFAULT_UNFAVORABLE_WEIGHT, regime_capital_weight

logger = logging.getLogger(__name__)


def regime_blend_weights(
    close: pd.Series,
    preferred: Sequence[frozenset[str]],
    *,
    unfavorable: float = DEFAULT_UNFAVORABLE_WEIGHT,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
) -> pd.DataFrame:
    """Побарні ваги капіталу [unfavorable, 1] для кожної суб-стратегії."""
    if not preferred:
        return pd.DataFrame(index=close.index)
    for pref in preferred:
        regime_capital_weight(pref, "range", "normal", unfavorable=unfavorable)
    state = named_market_state(close, trend_threshold=trend_threshold)
    structure = state["structure"]
    vol = state["vol"]
    parts: list[pd.Series] = []
    for i, pref in enumerate(preferred):
        weight = pd.Series(1.0, index=close.index)
        struct_tags = pref & STRUCTURE_LABELS
        vol_tags = pref & VOL_LABELS
        if struct_tags:
            weight = weight.where(structure.isin(struct_tags), unfavorable)
        if vol_tags:
            weight = weight * np.where(vol.isin(vol_tags), 1.0, unfavorable)
        parts.append(weight.rename(f"w{i}"))
    return pd.concat(parts, axis=1)


def regime_blend_signals(
    sig_df: pd.DataFrame,
    close: pd.Series,
    preferred: Sequence[frozenset[str]],
    *,
    unfavorable: float = DEFAULT_UNFAVORABLE_WEIGHT,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
) -> pd.Series:
    """М'яке зважування капіталу: position = Σ (w_i / n) · sig_i.

    Не quorum і не hard-switch. Якщо всі w=1 — збігається з mode='mean'.
    У несприятливому режимі частка стратегії стискається до unfavorable/n.
    """
    n = sig_df.shape[1]
    if n == 0:
        return pd.Series(0.0, index=sig_df.index)
    aligned = sig_df.reindex(close.index).fillna(0.0)
    w_df = regime_blend_weights(
        close,
        preferred,
        unfavorable=unfavorable,
        trend_threshold=trend_threshold,
    )
    w_df.columns = aligned.columns
    blended = (aligned * w_df).sum(axis=1) / n
    return blended.clip(-1.0, 1.0)


class EnsembleStrategy(Strategy):
    """Ensemble кількох стратегій (Portfolio Construction).

    Комбінує сигнали кількох суб-стратегій:
        - 'mean'   — усереднення (за замовчуванням);
        - 'vote'   — входимо лише при згоді всіх;
        - 'hedge'  — онлайн-зважування Hedge/EWA за історією прибутковостей
          (Gofer 2014, Ch.2): ваги адаптуються до концепт-дрейфу без
          перетренування; сигнал = Σ p_i,t · sig_i,t;
        - 'regime' — м'які ваги капіталу за family/preferred_regimes
          (Narang гл. 3/6: змішуємо капітал, не голоси).
    """

    name = "ensemble"
    family = "meta"
    preferred_regimes = frozenset()
    param_space: dict[str, tuple[float, float, float]] = {}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        strat_names = str(self.get("strategies", "mean_reversion,ml_strategy")).split(",")
        self.mode = str(self.get("mode", "mean"))  # "mean" | "vote" | "hedge" | "regime"

        from scalper_hft.strategies import get_strategy

        self.strats: list[Strategy] = []
        for s in strat_names:
            s = s.strip()
            if s:
                # Можна передати параметри через формат name:param1=val1:param2=val2
                parts = s.split(":")
                name = parts[0]
                p: dict[str, Any] = {}
                for part in parts[1:]:
                    if "=" in part:
                        k, v = part.split("=")
                        try:
                            p[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            p[k] = v
                self.strats.append(get_strategy(name, **p))

        self.needs_trades = any(s.needs_trades for s in self.strats)
        self.needs_funding = any(s.needs_funding for s in self.strats)

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        if not self.strats:
            return pd.Series(0.0, index=df.index)

        signals = []
        import inspect

        for s in self.strats:
            try:
                sig_params = inspect.signature(s.generate_signals).parameters
                kwargs = {}
                if "trades" in sig_params:
                    kwargs["trades"] = trades
                if "funding" in sig_params:
                    kwargs["funding"] = funding

                sig = s.generate_signals(df, **kwargs)
                signals.append(sig.fillna(0.0))
            except Exception as e:
                logger.warning("Стратегія %s помилка в ensemble: %s", s.name, e)
                signals.append(pd.Series(0.0, index=df.index))

        sig_df = pd.concat(signals, axis=1)

        if self.mode == "hedge":
            from scalper_hft.strategies.blend import hedge_blend_signals

            return hedge_blend_signals(sig_df, df["close"])

        if self.mode == "regime":
            preferred = [frozenset(s.preferred_regimes) for s in self.strats]
            unfavorable = float(self.get("unfavorable_weight", DEFAULT_UNFAVORABLE_WEIGHT))
            trend_threshold = float(self.get("trend_threshold", DEFAULT_TREND_THRESHOLD))
            return regime_blend_signals(
                sig_df,
                df["close"],
                preferred,
                unfavorable=unfavorable,
                trend_threshold=trend_threshold,
            )

        if self.mode == "vote":
            # Входимо тільки якщо всі стратегії мають однаковий знак (або більшість)
            # Тут проста логіка: якщо всі > 0, то 1. Якщо всі < 0, то -1. Інакше 0.
            def vote(row: pd.Series) -> float:
                if (row > 0).all():
                    return 1.0
                elif (row < 0).all():
                    return -1.0
                return 0.0

            return sig_df.apply(vote, axis=1)

        # За замовчуванням (mean): усереднюємо капітал (наприклад 0.5 від першої, 0.5 від другої)
        return sig_df.mean(axis=1)
