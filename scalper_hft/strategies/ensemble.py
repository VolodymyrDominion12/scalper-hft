from __future__ import annotations

from typing import Any
import pandas as pd
import logging

from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)


class EnsembleStrategy(Strategy):
    """Ensemble кількох стратегій (Portfolio Construction).

    Комбінує сигнали кількох суб-стратегій через усереднення або голосування.
    Зменшує дисперсію результату (Portfolio Construction, Narang).
    """

    name = "ensemble"
    param_space: dict[str, tuple[float, float, float]] = {}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        strat_names = str(self.get("strategies", "mean_reversion,ml_strategy")).split(",")
        self.mode = str(self.get("mode", "mean"))  # "mean" або "vote"

        from scalper_hft.strategies import get_strategy

        self.strats: list[Strategy] = []
        for s in strat_names:
            s = s.strip()
            if s:
                # Можна передати параметри через формат name:param1=val1:param2=val2
                parts = s.split(":")
                name = parts[0]
                p = {}
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
