"""HMM-гейтований mean-reversion скальпер (alpha-гіпотеза, Спринт 4).

Гіпотеза: mean reversion працює ЛИШЕ у «спокійному» режимі ринку (низька
реалізована волатильність, без тренду). HMM (FSPML Ch.4.5) розділяє ринок
на стани за [ret, |ret|, vol]; входимо в реверсію лише коли поточна
фільтрована ймовірність «спокійного» стану ≥ hmm_threshold.

Без lookahead:
    - HMM навчається лише на ПЕРШИХ `hmm_fit_bars` барах (фіксована модель);
    - на решті ряду використовуються ФІЛЬТРОВАНІ ймовірності стану
      (forward-only, `GaussianHMM.filtered_proba`) — жодних майбутніх даних;
    - сигнал на закритті бару t → виконання з t+1 (рушій).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scalper_hft.strategies.base import Strategy
from scalper_hft.strategies.mean_reversion import MeanReversionScalper

logger = logging.getLogger(__name__)

try:
    from scalper_hft.features.hmm_regime import GaussianHMM

    _HAS_HMM = True
except ImportError:  # pragma: no cover
    GaussianHMM = None  # type: ignore
    _HAS_HMM = False


class HmmReversionScalper(Strategy):
    """Mean reversion з HMM-гейтом режиму (спокійний стан = реверсія).

    Параметри:
        rsi_period/oversold/overbought/bb_period/min_atr_pct/stop_atr_mult/max_trend
            — ті самі, що в MeanReversionScalper;
        hmm_states     : кількість HMM-станів (default 3);
        hmm_threshold  : мін. P(спокійний стан) для входу (default 0.5);
        hmm_fit_bars   : скільки перших барів для навчання HMM (default 2000).
    """

    name = "hmm_reversion"
    family = "mean_reversion"
    preferred_regimes = frozenset({"range", "low", "normal"})
    needs_trades = False

    param_space: dict[str, tuple[float, float, float]] = {
        "rsi_period": (5.0, 30.0, 1.0),
        "oversold": (10.0, 40.0, 2.0),
        "overbought": (60.0, 90.0, 2.0),
        "bb_period": (10.0, 60.0, 5.0),
        "min_atr_pct": (0.0005, 0.005, 0.0005),
        "max_trend": (0.2, 1.0, 0.1),
        "hmm_threshold": (0.3, 0.8, 0.1),
    }

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        bb_period: int = 20,
        min_atr_pct: float = 0.001,
        stop_atr_mult: float = 2.0,
        max_trend: float = 0.6,
        hmm_states: int = 3,
        hmm_threshold: float = 0.5,
        hmm_fit_bars: int = 2000,
        skip_high_vol: bool = True,
    ) -> None:
        super().__init__(
            rsi_period=int(rsi_period),
            oversold=oversold,
            overbought=overbought,
            bb_period=int(bb_period),
            min_atr_pct=min_atr_pct,
            stop_atr_mult=stop_atr_mult,
            max_trend=max_trend,
            hmm_states=int(hmm_states),
            hmm_threshold=hmm_threshold,
            hmm_fit_bars=int(hmm_fit_bars),
            skip_high_vol=skip_high_vol,
        )

    def _calm_state_mask(self, close: pd.Series, n_states: int, fit_bars: int, threshold: float) -> pd.Series:
        """Маска «спокійного» HMM-стану (каузальна, без lookahead).

        Returns: Series bool, index = close.index.
        """
        if not _HAS_HMM:
            logger.warning("HMM недоступний — гейт вимкнено")
            return pd.Series(True, index=close.index)

        ret = close.pct_change().fillna(0.0)
        vol = ret.rolling(20, min_periods=10).std().fillna(0.0)
        obs = pd.DataFrame({"ret": ret, "abs_ret": ret.abs(), "vol": vol}).iloc[20:]
        if len(obs) < n_states * 20:
            return pd.Series(True, index=close.index)

        fit = obs.iloc[: min(fit_bars, len(obs))].values
        try:
            model = GaussianHMM(n_states=n_states, seed=42).fit(fit)
        except ValueError:
            return pd.Series(True, index=close.index)

        if model.covars_ is None:
            return pd.Series(True, index=close.index)

        # «спокійний» стан = мінімальна дисперсія на vol-фічі (індекс 2)
        calm = int(np.argmin(model.covars_[:, 2]))
        post = model.filtered_proba(obs.values)
        p_calm = post[:, calm]
        mask = pd.Series(p_calm >= threshold, index=obs.index)
        return mask.reindex(close.index).fillna(False).astype(bool)

    def _base_scalper(self) -> MeanReversionScalper:
        """Базовий mean-reversion скальпер з поточними параметрами."""
        return MeanReversionScalper(
            rsi_period=int(self.get("rsi_period", 14)),
            oversold=self.get("oversold", 30.0),
            overbought=self.get("overbought", 70.0),
            bb_period=int(self.get("bb_period", 20)),
            min_atr_pct=self.get("min_atr_pct", 0.001),
            stop_atr_mult=self.get("stop_atr_mult", 2.0),
            max_trend=self.get("max_trend", 0.6),
            skip_high_vol=bool(self.get("skip_high_vol", True)),
        )

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        # базові mean-reversion сигнали (перевикористовуємо MeanReversionScalper)
        signals = self._base_scalper().generate_signals(df)

        # HMM-гейт: лише у «спокійному» стані
        calm = self._calm_state_mask(
            df["close"],
            n_states=int(self.get("hmm_states", 3)),
            fit_bars=int(self.get("hmm_fit_bars", 2000)),
            threshold=float(self.get("hmm_threshold", 0.5)),
        )
        return signals.where(calm, other=0)

    def exit_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рівні SL/TP — ті самі, що в MeanReversionScalper (ціль = середина BB)."""
        return self._base_scalper().exit_levels(df)
