"""MLStrategy — alpha-модель на основі LightGBM з AFML pipeline.

Архітектура (Inside the Black Box, Narang):
    Data → Features (frac_diff + micro) → Triple-Barrier Label → LightGBM
    → сигнал +1/-1, опціонально з confidence-фільтром, sizing із імовірностей
    та повним мета-лейблінгом (AFML Ch.3.6–3.7, Ch.10.3).

Walk-forward схема:
    Модель навчається на `train_bars` барах до поточного моменту,
    генерує сигнали на наступних `test_bars` (OOS), потім перенавчається.
    Це еквівалентно expanding/rolling window у реальній торгівлі.

Особливості:
    - sample_weight = uniqueness + time-decay (AFML Ch.4)
    - frac_diff фічі для збереження пам'яті ряду (AFML Ch.5)
    - regime_filter: відключає торгівлю у несприятливих режимах
    - confidence_thr: сигнал 0 при впевненості < порога (працює через proba)
    - prob_size: розмір позиції ∝ |prob_to_size(p)| (AFML Ch.10.3)
    - meta_filter: повний мета-лейблінг — primary задає сторону, мета-модель
      задає розмір P(meta=1) (AFML Ch.3.7 + Ch.10.3)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.features.regimes import trend_strength, volatility_regime
from scalper_hft.ml.bet_sizing import meta_size, prob_to_size
from scalper_hft.ml.features import build_labeled_dataset
from scalper_hft.ml.trainer import MlResult, train_walk_forward, train_walk_forward_meta
from scalper_hft.strategies.base import Strategy

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier  # type: ignore

    _HAS_LGBM = True
except ImportError:
    LGBMClassifier = None  # type: ignore
    _HAS_LGBM = False


class MLStrategy(Strategy):
    """LightGBM alpha-модель з Triple-Barrier labeling та AFML sample weights.

    Параметри (передаються через **params):
        train_bars      : кількість барів у rolling train-вікні (default 2000)
        test_bars       : кількість барів на OOS крок (default 500)
        pt              : profit-take множник × ATR (default 1.0)
        sl              : stop-loss множник × ATR (default 1.0)
        holding_bars    : вертикальний бар'єр (default 10)
        decay           : time-decay для ваг (default 0.9)
        frac_d          : ступінь fractional diff (default 0.4)
        confidence_thr  : мін. впевненість для сигналу (default 0.50 = без фільтра)
        regime_filter   : 'none' | 'vol' | 'trend' (default 'none')
        vol_regime_ok   : 'low,normal' | 'normal' | 'high,normal' (default 'normal')
        prob_size       : sizing ∝ впевненості primary (AFML Ch.10.3), default False
        meta_filter     : повний мета-лейблінг + sizing (default False)
        meta_scale      : множник розміру мета-ставки (default 1.0)
        ood_threshold   : DI veto (0 = вимкнено); >0 блокує OOD-бари
    """

    name = "ml_strategy"
    needs_trades = False

    param_space: dict[str, tuple[float, float, float]] = {
        "pt": (0.5, 2.0, 0.25),
        "sl": (0.5, 2.0, 0.25),
        "holding_bars": (5, 30, 5),
        "decay": (0.7, 1.0, 0.1),
        "confidence_thr": (0.50, 0.65, 0.025),
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._model: LGBMClassifier | None = None
        self._last_trained_at: int = -1  # індекс останнього рефіту
        self._oos_preds: pd.Series | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Генерує сигнали позиції [-1, 0, +1] (або безперервні у [-1, 1]).

        Walk-forward: модель навчається на перших `train_bars` барах,
        далі прогнозує rolling кроками по `test_bars`.

        Returns:
            Series позицій у [-1, 1] з індексом df.index.
        """
        if not _HAS_LGBM:
            raise ImportError("Встановіть lightgbm: uv add --optional ml lightgbm scikit-learn")

        train_bars = int(self.get("train_bars", 2000))
        test_bars = int(self.get("test_bars", 500))
        pt = float(self.get("pt", 1.0))
        sl = float(self.get("sl", 1.0))
        holding_bars = int(self.get("holding_bars", 10))
        decay = float(self.get("decay", 0.9))
        frac_d = float(self.get("frac_d", 0.4))
        confidence_thr = float(self.get("confidence_thr", 0.5))
        regime_filter = str(self.get("regime_filter", "none"))
        vol_regime_ok = str(self.get("vol_regime_ok", "normal")).split(",")
        prob_size = bool(self.get("prob_size", False))
        meta_filter = bool(self.get("meta_filter", False))
        meta_scale = float(self.get("meta_scale", 1.0))
        ood_threshold = float(self.get("ood_threshold", 0.0))
        add_hmm = bool(self.get("add_hmm", False))
        add_garch = bool(self.get("add_garch", False))
        hmm_states = int(self.get("hmm_states", 3))

        # Будуємо labeled dataset
        try:
            X, y, w = build_labeled_dataset(
                df=df,
                trades=trades,
                mode="triple_barrier",
                pt=pt,
                sl=sl,
                holding_bars=holding_bars,
                decay=decay,
                frac_d=frac_d,
                add_frac_diff=True,
                add_hmm=add_hmm,
                add_garch=add_garch,
                hmm_states=hmm_states,
            )
        except ValueError as e:
            logger.warning("MLStrategy: %s — повертаю нульові сигнали", e)
            return pd.Series(0, index=df.index)

        if len(X) < train_bars + test_bars:
            logger.warning(
                "MLStrategy: замало зразків (%d < %d). Збільшіть датасет.",
                len(X),
                train_bars + test_bars,
            )
            return pd.Series(0, index=df.index)

        if meta_filter:
            # ── Повний мета-лейблінг: primary → сторона, мета → розмір ──
            side, p_meta = train_walk_forward_meta(
                X=X,
                y=y,
                train_size=train_bars,
                test_size=test_bars,
                sample_weights=w,
            )
            size = meta_size(p_meta.values) * meta_scale
            signals = side.astype(float) * size
        else:
            # ── Primary walk-forward ──
            result = train_walk_forward(
                X=X,
                y=y,
                train_size=train_bars,
                test_size=test_bars,
                sample_weights=w,
                close=df["close"],
            )
            side = result.predictions  # {-1, +1} на OOS-індексах
            p_side = result.probabilities  # P(клас +1)

            # Confidence-фільтр через proba (раніше був no-op без proba)
            side = self._apply_confidence_filter(side, p_side, confidence_thr)

            if prob_size and p_side is not None:
                size = np.abs(prob_to_size(p_side.values))
                signals = side.astype(float) * size
            else:
                signals = side.astype(float)

        if ood_threshold > 0:
            from scalper_hft.ml.ood import apply_ood_veto, fit_ood_stats, ood_mask

            train_end = min(train_bars, len(X))
            mean, std = fit_ood_stats(X.iloc[:train_end])
            in_dist = ood_mask(X, mean, std, ood_threshold)
            ood_series = pd.Series(in_dist, index=X.index)
            signals = apply_ood_veto(signals, ood_series.reindex(signals.index, fill_value=True))

        # Режимний фільтр
        if regime_filter == "vol":
            regime = volatility_regime(df["close"])
            ok = regime.isin(vol_regime_ok).reindex(signals.index, fill_value=False)
            signals = signals.where(ok, other=0.0)
        elif regime_filter == "trend":
            ts = trend_strength(df["close"])
            # торгуємо лише коли тренд слабкий (mean-reversion умови)
            ok = (ts < 0.3).reindex(signals.index, fill_value=False)
            signals = signals.where(ok, other=0.0)

        # Вирівнюємо на весь df.index (де немає прогнозу → 0)
        full = pd.Series(0.0, index=df.index, dtype=float)
        full.update(signals.astype(float))
        return full

    def get_last_result(self) -> MlResult | None:
        """Повертає останній MlResult (для аналізу)."""
        return getattr(self, "_last_result", None)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_confidence_filter(side: pd.Series, p_side: pd.Series | None, threshold: float) -> pd.Series:
        """При threshold > 0.5 залишаємо лише 'впевнені' сигнали.

        Впевненість = max(p, 1−p). Без proba (p_side=None) — повертаємо як є.
        """
        if p_side is None or threshold <= 0.5:
            return side
        conf = pd.concat([p_side, 1.0 - p_side], axis=1).max(axis=1)
        return side.where(conf >= threshold, other=0)

    def exit_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рівні SL/TP за логікою triple-barrier labeling (ml/labeling.py).

        Бар'єри відносні close бару: tp = close ± pt·σ, sl = close ∓ sl·σ,
        σ = `_daily_vol(close, span=100)` — той самий волатильнісний таргет,
        що використовується при побудові навчальних міток (label_from_ohlcv).
        """
        from scalper_hft.ml.labeling import _daily_vol

        close = df["close"]
        sigma = _daily_vol(close, span=100).fillna(0.0)
        pt = float(self.get("pt", 1.0))
        sl = float(self.get("sl", 1.0))
        out = pd.DataFrame(index=df.index, dtype=float)
        out["tp_long"] = close + pt * sigma
        out["sl_long"] = close - sl * sigma
        out["tp_short"] = close - pt * sigma
        out["sl_short"] = close + sl * sigma
        return out
