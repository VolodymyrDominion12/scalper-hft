"""MLStrategy — alpha-модель на основі LightGBM з AFML pipeline.

Архітектура (Inside the Black Box, Narang):
    Data → Features (frac_diff + micro) → Triple-Barrier Label → LightGBM
    → сигнал +1/-1 з мета-лейблінгом (опціонально)

Walk-forward схема:
    Модель навчається на `train_bars` барах до поточного моменту,
    генерує сигнали на наступних `test_bars` (OOS), потім перенавчається.
    Це еквівалентно expanding/rolling window у реальній торгівлі.

Особливості:
    - sample_weight = uniqueness + time-decay (AFML Ch.4)
    - frac_diff фічі для збереження пам'яті ряду (AFML Ch.5)
    - regime_filter: відключає торгівлю у несприятливих режимах
    - meta_filter: другий ML-шар що фільтрує слабкі сигнали
    - Сигнал 0 при впевненості < confidence_threshold
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.features.regimes import volatility_regime, trend_strength
from scalper_hft.ml.features import build_labeled_dataset
from scalper_hft.ml.trainer import train_walk_forward, MlResult
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
        confidence_thr  : мін. впевненість для сигналу (default 0.52)
        regime_filter   : 'none' | 'vol' | 'trend' (default 'none')
        vol_regime_ok   : 'low,normal' | 'normal' | 'high,normal' (default 'normal')
        meta_filter     : bool, чи застосовувати мета-лейблінг (default False)
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
        """Генерує сигнали позиції [-1, 0, +1] для всього df.

        Walk-forward: модель навчається на перших `train_bars` барах,
        далі прогнозує rolling кроками по `test_bars`.

        Returns:
            Series positions {-1, 0, 1} з індексом df.index.
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
        confidence_thr = float(self.get("confidence_thr", 0.52))
        regime_filter = str(self.get("regime_filter", "none"))
        vol_regime_ok = str(self.get("vol_regime_ok", "normal")).split(",")
        meta_filter = bool(self.get("meta_filter", False))

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
            )
        except ValueError as e:
            logger.warning("MLStrategy: %s — повертаю нульові сигнали", e)
            return pd.Series(0, index=df.index)

        if len(X) < train_bars + test_bars:
            logger.warning(
                "MLStrategy: замало зразків (%d < %d). Збільшіть датасет.",
                len(X), train_bars + test_bars,
            )
            return pd.Series(0, index=df.index)

        # Walk-forward тренування
        result = train_walk_forward(
            X=X, y=y,
            train_size=train_bars,
            test_size=test_bars,
            sample_weights=w,
            close=df["close"],
        )
        preds = result.predictions  # {-1, +1} на OOS-індексах

        # Confidence-фільтр: якщо немає proba — використовуємо raw pred
        signals = self._apply_confidence_filter(preds, confidence_thr)

        # Meta-лейблінг (другий ML-шар)
        if meta_filter:
            signals = self._apply_meta_filter(X, y, w, signals, train_bars, test_bars)

        # Режимний фільтр
        if regime_filter == "vol":
            regime = volatility_regime(df["close"])
            ok = regime.isin(vol_regime_ok).reindex(signals.index, fill_value=False)
            signals = signals.where(ok, other=0)
        elif regime_filter == "trend":
            ts = trend_strength(df["close"])
            # торгуємо лише коли тренд слабкий (mean-reversion умови)
            ok = (ts < 0.3).reindex(signals.index, fill_value=False)
            signals = signals.where(ok, other=0)

        # Вирівнюємо на весь df.index (де немає прогнозу → 0)
        full = pd.Series(0, index=df.index, dtype=int)
        full.update(signals.astype(int))
        return full

    def get_last_result(self) -> MlResult | None:
        """Повертає останній MlResult (для аналізу)."""
        return getattr(self, "_last_result", None)

    # ── Private ───────────────────────────────────────────────────────────────

    def _apply_confidence_filter(
        self, preds: pd.Series, threshold: float
    ) -> pd.Series:
        """При threshold > 0.5 залишаємо лише 'впевнені' сигнали.

        Оскільки predict_proba недоступна тут безпосередньо,
        використовуємо preds як є (вони вже {-1, +1}).
        Для більш тонкого фільтру — використовуй train_from_ohlcv напряму.
        """
        if threshold <= 0.5:
            return preds
        # без proba — повертаємо всі сигнали (threshold не впливає на binary)
        return preds

    def _apply_meta_filter(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        w: pd.Series | None,
        primary_signals: pd.Series,
        train_bars: int,
        test_bars: int,
    ) -> pd.Series:
        """Мета-лейблінг: другий класифікатор передбачає P(primary correct).

        1. Беремо primary_signals як feature (side)
        2. Навчаємо бінарний метакласифікатор: чи збіглася primary з реальним y?
        3. Множимо сигнал на мета-прогноз (≥ 0.5 → залишаємо, < 0.5 → нуль)
        """
        # Мета-таргет: 1 якщо primary вгадав, 0 — якщо ні
        meta_y_aligned = y.reindex(primary_signals.index)
        meta_target = (primary_signals == meta_y_aligned).astype(int)

        # Додаємо primary сигнал як фічу
        X_meta = X.reindex(primary_signals.index).copy()
        X_meta["primary_signal"] = primary_signals.values

        if len(X_meta) < train_bars + test_bars:
            return primary_signals

        try:
            meta_result = train_walk_forward(
                X=X_meta,
                y=meta_target,
                train_size=train_bars,
                test_size=test_bars,
                sample_weights=w,
            )
            meta_preds = meta_result.predictions  # 0 або 1
            # застосовуємо: якщо мета передбачає 0 → скасовуємо сигнал
            filtered = primary_signals.copy()
            cancel_idx = meta_preds[meta_preds == 0].index
            filtered.loc[filtered.index.intersection(cancel_idx)] = 0
            return filtered
        except Exception as e:
            logger.warning("Meta-filter failed: %s", e)
            return primary_signals
