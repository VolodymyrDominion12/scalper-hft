"""Єдиний детектор режиму ринку: HMM + rule-based → RegimeState.

Проблема, яку вирішує цей модуль: раніше кожна стратегія визначала режим
окремо (hmm_reversion — власний HMM, ensemble — rule-based), що призводило
до неузгоджених рішень. RegimeDetector — єдине джерело «правди» про ринок
для RegimeSupervisor та всіх стратегій.

Два режими роботи:
    batch (detect):  для бектесту — навчає HMM на перших fit_bars, далі
                     filtered_proba (forward-only, без lookahead).
    online (step):   для live — інкрементально оновлює rule-based фічі,
                     HMM вже навчений і не змінюється.

Архітектура:
    RegimeState — dataclass з поточним станом ринку.
    RegimeDetector — head: навчання + batch/online inference.

Без lookahead:
    - HMM навчається ТІЛЬКИ на перших fit_bars барах.
    - Далі використовується filtered_proba (forward-only alpha).
    - Rule-based (EMA, vol) — каузальні за конструкцією.
    - Сигнал на закритті бару t → виконання з бару t+1 (рушій).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.features.hmm_regime import GaussianHMM
from scalper_hft.features.regimes import (
    DEFAULT_TREND_THRESHOLD,
    named_market_state,
    volatility_regime,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# RegimeState
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RegimeState:
    """Поточний стан ринку (snapshot одного бару).

    Attributes:
        structure:   rule-based структура: "range" | "trend_up" | "trend_down".
        vol:         rule-based волатильність: "low" | "normal" | "high".
        label:       складений ярлик "structure|vol", напр. "range|low".
        hmm_state:   індекс найімовірнішого HMM-стану (0..n_states-1).
                     -1 якщо HMM ще не навчений.
        hmm_probs:   вектор P(режим=k) за HMM (довжина n_states).
                     порожній масив якщо HMM не навчений.
        confidence:  max(hmm_probs) — впевненість HMM у поточному стані.
    """

    structure: str
    vol: str
    label: str
    hmm_state: int
    hmm_probs: np.ndarray
    confidence: float

    def is_trending(self) -> bool:
        return self.structure in ("trend_up", "trend_down")

    def is_range(self) -> bool:
        return self.structure == "range"

    def is_high_vol(self) -> bool:
        return self.vol == "high"

    def is_low_vol(self) -> bool:
        return self.vol == "low"

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "vol": self.vol,
            "label": self.label,
            "hmm_state": self.hmm_state,
            "confidence": self.confidence,
        }


# ────────────────────────────────────────────────────────────────────────────
# RegimeDetector
# ────────────────────────────────────────────────────────────────────────────


class RegimeDetector:
    """Єдиний детектор режиму: HMM + rule-based features.

    Параметри:
        n_hmm_states:   кількість прихованих станів HMM (default 3).
        hmm_fit_bars:   скільки перших барів використовувати для навчання HMM.
                        Більше = краща модель, але більше прогріву. (default 2000)
        ema_fast:       швидка EMA для trend_strength (default 9).
        ema_slow:       повільна EMA для trend_strength (default 50).
        trend_threshold: поріг відстані EMA для класифікації тренду (default 0.35).
        vol_lookback:   вікно реалізованої волатильності в барах (default 60).
        vol_percentile_window: вікно для процентилю vol (default 500).
        hmm_seed:       seed для відтворюваності HMM (default 42).
    """

    def __init__(
        self,
        n_hmm_states: int = 3,
        hmm_fit_bars: int = 2000,
        ema_fast: int = 9,
        ema_slow: int = 50,
        trend_threshold: float = DEFAULT_TREND_THRESHOLD,
        vol_lookback: int = 60,
        vol_percentile_window: int = 500,
        hmm_seed: int = 42,
    ) -> None:
        self.n_hmm_states = n_hmm_states
        self.hmm_fit_bars = hmm_fit_bars
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.trend_threshold = trend_threshold
        self.vol_lookback = vol_lookback
        self.vol_percentile_window = vol_percentile_window
        self.hmm_seed = hmm_seed

        # Стан після fit()
        self._hmm: GaussianHMM | None = None
        self._fitted: bool = False

        # Інкрементальний стан для live (step())
        self._close_buf: list[float] = []
        self._last_state: RegimeState | None = None

    # ── Batch (бектест) ─────────────────────────────────────────────────────

    def fit(self, close: pd.Series) -> RegimeDetector:
        """Навчає HMM на перших hmm_fit_bars барах.

        КРИТИЧНО: навчання лише на ранньому відрізку — без lookahead на решту ряду.
        Метод мутує self (зберігає навчену модель). Поверненим об'єктом можна
        одразу викликати detect().
        """
        ret = close.pct_change().fillna(0.0)
        vol = ret.rolling(20, min_periods=10).std().fillna(0.0)
        obs = pd.DataFrame({"ret": ret, "abs_ret": ret.abs(), "vol": vol}).iloc[20:]

        n_fit = min(self.hmm_fit_bars, len(obs))
        if n_fit < self.n_hmm_states * 10:
            logger.warning(
                "RegimeDetector: замало даних для HMM (%d барів < %d × 10). "
                "HMM вимкнено.",
                n_fit,
                self.n_hmm_states,
            )
            self._fitted = False
            return self

        X_fit = obs.iloc[:n_fit].values
        try:
            self._hmm = GaussianHMM(
                n_states=self.n_hmm_states,
                seed=self.hmm_seed,
            ).fit(X_fit)
            self._fitted = True
            logger.info(
                "RegimeDetector: HMM навчений на %d барах (%d станів).",
                n_fit,
                self.n_hmm_states,
            )
        except ValueError as exc:
            logger.warning("RegimeDetector: HMM не навчився: %s. HMM вимкнено.", exc)
            self._fitted = False

        return self

    def detect(self, close: pd.Series) -> pd.DataFrame:
        """Batch-детекція режиму для всього ряду (для бектесту).

        Якщо fit() ще не викликано — викликає автоматично.

        Повертає DataFrame з колонками:
            structure, vol, label, hmm_state, hmm_p0..hmm_pK-1, confidence.
        Індексований як close.index.

        Без lookahead:
            - rule-based (EMA, vol) — каузальні.
            - HMM: filtered_proba (forward-only) на ВСЬОМУ ряді з ФІКСОВАНОЮ моделлю.
        """
        if not self._fitted:
            self.fit(close)

        # Rule-based
        state_df = named_market_state(
            close,
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            trend_threshold=self.trend_threshold,
            vol_lookback=self.vol_lookback,
            vol_percentile_window=self.vol_percentile_window,
        )

        # HMM filtered proba (без lookahead)
        hmm_cols: dict[str, pd.Series] = {}
        if self._fitted and self._hmm is not None:
            ret = close.pct_change().fillna(0.0)
            vol = ret.rolling(20, min_periods=10).std().fillna(0.0)
            obs = pd.DataFrame({"ret": ret, "abs_ret": ret.abs(), "vol": vol}).iloc[20:]
            X = obs.values

            try:
                post = self._hmm.filtered_proba(X)  # (T, K) — forward-only
                states = np.argmax(post, axis=1)
                confidence = post[np.arange(len(post)), states]

                state_df.loc[obs.index, "hmm_state"] = states.astype(float)
                state_df.loc[obs.index, "confidence"] = confidence

                for k in range(self.n_hmm_states):
                    col = f"hmm_p{k}"
                    state_df.loc[obs.index, col] = post[:, k]
                    hmm_cols[col] = state_df[col]

            except Exception as exc:
                logger.warning("RegimeDetector.detect: HMM inference помилка: %s", exc)

        # Заповнення відсутніх колонок
        for col in ["hmm_state", "confidence"] + list(hmm_cols.keys()):
            if col not in state_df.columns:
                state_df[col] = 0.0

        state_df["hmm_state"] = state_df["hmm_state"].ffill().fillna(-1).astype(int)
        state_df["confidence"] = state_df["confidence"].ffill().fillna(0.0)
        state_df = state_df.fillna(0.0)
        state_df["hmm_state"] = state_df["hmm_state"].astype(int)

        return state_df

    def detect_states(self, close: pd.Series) -> list[RegimeState]:
        """Детекція → список RegimeState (один об'єкт на бар)."""
        df = self.detect(close)
        hmm_p_cols = [f"hmm_p{k}" for k in range(self.n_hmm_states) if f"hmm_p{k}" in df.columns]
        states: list[RegimeState] = []
        for _, row in df.iterrows():
            probs = np.array([float(row.get(c, 0.0)) for c in hmm_p_cols])
            states.append(
                RegimeState(
                    structure=str(row["structure"]),
                    vol=str(row["vol"]),
                    label=str(row["label"]),
                    hmm_state=int(row["hmm_state"]),
                    hmm_probs=probs,
                    confidence=float(row["confidence"]),
                )
            )
        return states

    # ── Online (live) ────────────────────────────────────────────────────────

    def step(self, new_close: float) -> RegimeState:
        """Інкрементальне оновлення режиму для live (один бар).

        Буфер накопичує закриті ціни. Коли накопичено достатньо даних —
        автоматично навчає HMM (один раз). Повертає RegimeState поточного бару.
        """
        self._close_buf.append(float(new_close))

        # Автоматичне навчання HMM коли є достатньо даних
        if not self._fitted and len(self._close_buf) >= self.hmm_fit_bars:
            close_series = pd.Series(self._close_buf, dtype=float)
            self.fit(close_series)

        # Потрібно мінімум ema_slow + vol_lookback + 20 барів для стабільних фіч
        warmup = max(self.ema_slow + 20, self.vol_lookback + 20)
        if len(self._close_buf) < warmup:
            state = RegimeState(
                structure="range",
                vol="normal",
                label="range|normal",
                hmm_state=-1,
                hmm_probs=np.zeros(self.n_hmm_states),
                confidence=0.0,
            )
            self._last_state = state
            return state

        # Rule-based: достатньо останніх max(ema_slow, vol_percentile_window) барів
        buf_size = max(self.ema_slow * 3, self.vol_lookback + 20, 600)
        recent = pd.Series(self._close_buf[-buf_size:], dtype=float)
        state_df = named_market_state(
            recent,
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            trend_threshold=self.trend_threshold,
            vol_lookback=self.vol_lookback,
            vol_percentile_window=min(self.vol_percentile_window, len(recent) // 2),
        )
        last_row = state_df.iloc[-1]
        structure = str(last_row["structure"])
        vol = str(last_row["vol"])
        label = str(last_row["label"])

        # HMM inference
        hmm_state = -1
        hmm_probs = np.zeros(self.n_hmm_states)
        confidence = 0.0

        if self._fitted and self._hmm is not None:
            ret_s = recent.pct_change().fillna(0.0)
            vol_s = ret_s.rolling(20, min_periods=10).std().fillna(0.0)
            obs = pd.DataFrame({"ret": ret_s, "abs_ret": ret_s.abs(), "vol": vol_s}).iloc[20:]
            if len(obs) >= self.n_hmm_states:
                try:
                    post = self._hmm.filtered_proba(obs.values)
                    hmm_probs = post[-1]  # лише останній бар
                    hmm_state = int(np.argmax(hmm_probs))
                    confidence = float(hmm_probs[hmm_state])
                except Exception as exc:
                    logger.debug("RegimeDetector.step: HMM inference: %s", exc)

        state = RegimeState(
            structure=structure,
            vol=vol,
            label=label,
            hmm_state=hmm_state,
            hmm_probs=hmm_probs,
            confidence=confidence,
        )
        self._last_state = state
        return state

    @property
    def last_state(self) -> RegimeState | None:
        """Останній обчислений стан (None до першого step())."""
        return self._last_state

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Зберегти навчену модель (pickle)."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "hmm": self._hmm,
                    "fitted": self._fitted,
                    "params": {
                        "n_hmm_states": self.n_hmm_states,
                        "hmm_fit_bars": self.hmm_fit_bars,
                        "ema_fast": self.ema_fast,
                        "ema_slow": self.ema_slow,
                        "trend_threshold": self.trend_threshold,
                        "vol_lookback": self.vol_lookback,
                        "vol_percentile_window": self.vol_percentile_window,
                        "hmm_seed": self.hmm_seed,
                    },
                },
                f,
            )
        logger.info("RegimeDetector збережений у %s", path)

    @classmethod
    def load(cls, path: str) -> RegimeDetector:
        """Завантажити збережену модель."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        det = cls(**data["params"])
        det._hmm = data["hmm"]
        det._fitted = data["fitted"]
        return det


__all__ = ["RegimeDetector", "RegimeState"]
