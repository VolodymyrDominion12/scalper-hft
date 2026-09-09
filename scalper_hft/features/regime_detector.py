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
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scalper_hft.features.hmm_regime import GaussianHMM
from scalper_hft.features.regimes import (
    DEFAULT_TREND_THRESHOLD,
    apply_min_dwell,
    htf_market_structure,
    named_market_state,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Funding skew + liquidity helpers (2A)
# ────────────────────────────────────────────────────────────────────────────


def _funding_skew_series(funding: pd.Series, window: int = 720) -> pd.Series:
    """Z-score funding-ставки (каузально): skew = (f - rolling_mean)/rolling_std.

    Додатнє значення = ринок платить longs (перегрітість long), від'ємне = short.
    `window` за замовчуванням ~30 днів для 1h. Повертає Series 0.0 якщо даних мало.
    """
    f = pd.Series(funding, dtype=float).sort_index()
    if len(f) < max(window, 20):
        return pd.Series(0.0, index=f.index)
    mu = f.rolling(window, min_periods=window // 2).mean()
    sd = f.rolling(window, min_periods=window // 2).std()
    z = (f - mu) / sd.replace(0.0, np.nan)
    return z.fillna(0.0).clip(-3.0, 3.0)


def _liquidity_label_from_volume(close: pd.Series, volume: pd.Series, window: int = 120) -> pd.Series:
    """Amihud-ілюіквідність → label "low"|"normal"|"high" (каузально).

    illiq_t = mean(|ret|/dollarvol) over rolling window. Високе illiq = low liquidity.
    Розбиття — по rolling-квантилю вікна (медіана + 1.5×IQR). Повертає "normal"
    на warmup.
    """
    if volume is None or len(volume) != len(close):
        return pd.Series("normal", index=close.index)
    ret = close.pct_change().fillna(0.0)
    dvol = (volume.astype(float) * close.astype(float)).clip(lower=1e-12)
    illiq = (ret.abs() / dvol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    roll = illiq.rolling(window, min_periods=window // 2)
    med = roll.median()
    iqr = (roll.quantile(0.75) - roll.quantile(0.25)).clip(lower=1e-15)
    high_thr = med + 1.5 * iqr
    low_thr = (med - 0.5 * iqr).clip(lower=0.0)
    label = pd.Series("normal", index=close.index)
    label = label.where(illiq <= high_thr, "low")  # висока illiq = low liquidity
    label = label.where(illiq >= low_thr, "high")
    return label.fillna("normal")


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
        funding_skew: z-score funding-ставки (2A): додатнє = премія longs,
                     від'ємне = премія shorts. 0.0 якщо funding недоступний.
        liquidity:   ліквідність-проксі (2A): "low" | "normal" | "high".
                     "normal" якщо дані недоступні.
    """

    structure: str
    vol: str
    label: str
    hmm_state: int
    hmm_probs: np.ndarray
    confidence: float
    funding_skew: float = 0.0
    liquidity: str = "normal"

    def is_trending(self) -> bool:
        return self.structure in ("trend_up", "trend_down")

    def is_range(self) -> bool:
        return self.structure == "range"

    def is_high_vol(self) -> bool:
        return self.vol == "high"

    def is_low_vol(self) -> bool:
        return self.vol == "low"

    @property
    def joint_label(self) -> str:
        """Складений ярлик усіх факторів (2A): structure|vol|hmm|funding|liq.

        Єдина «правда» про режим для роутингу: об'єднує structure × vol ×
        HMM-стан × знак funding-skew × liquidity. Використовується мета-шаром
        для posterior-weighted routing та validated regime→strategy map (2B).
        """
        fsign = "p" if self.funding_skew > 0.5 else ("n" if self.funding_skew < -0.5 else "0")
        return f"{self.label}|h{self.hmm_state}|f{fsign}|{self.liquidity}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "vol": self.vol,
            "label": self.label,
            "hmm_state": self.hmm_state,
            "confidence": self.confidence,
            "funding_skew": self.funding_skew,
            "liquidity": self.liquidity,
            "joint_label": self.joint_label,
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
        min_dwell_bars: гістерезис структури: новий structure-режим приймається
                        лише після N послідовних барів (default 0 = без змін).
                        Зменшує regime churn для мета-стратегій (дослідження:
                        min-dwell ~2 дні ріже churn на ~68%).
        htf_structure:  старший таймфрейм для structure (напр. "1d") замість
                        same-TF EMA-cross; None = локальна структура (default).
                        Каузально: лише закриті htf-вікна ≤ t (iter3: 1h-структура
                        запізнюється відносно тренду старшого ТФ).
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
        min_dwell_bars: int = 0,
        htf_structure: str | None = None,
    ) -> None:
        self.n_hmm_states = n_hmm_states
        self.hmm_fit_bars = hmm_fit_bars
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.trend_threshold = trend_threshold
        self.vol_lookback = vol_lookback
        self.vol_percentile_window = vol_percentile_window
        self.hmm_seed = hmm_seed
        self.min_dwell_bars = int(min_dwell_bars)
        self.htf_structure = htf_structure

        # Стан після fit()
        self._hmm: GaussianHMM | None = None
        self._fitted: bool = False

        # Інкрементальний стан для live (step())
        self._close_buf: list[float] = []
        self._funding_buf: list[float] = []  # funding-ставки (2A)
        self._volume_buf: list[float] = []  # обсяги (2A)
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
                "RegimeDetector: замало даних для HMM (%d барів < %d × 10). HMM вимкнено.",
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

    def detect(
        self,
        close: pd.Series,
        funding: pd.Series | None = None,
        volume: pd.Series | None = None,
        refit_every: int = 0,
    ) -> pd.DataFrame:
        """Batch-детекція режиму для всього ряду (для бектесту).

        Якщо fit() ще не викликано — викликає автоматично.

        Повертає DataFrame з колонками:
            structure, vol, label, hmm_state, hmm_p0..hmm_pK-1, confidence,
            funding_skew, liquidity, joint_label.
        Індексований як close.index.

        Без lookahead:
            - rule-based (EMA, vol) — каузальні.
            - HMM: filtered_proba (forward-only) на ВСЬОМУ ряді з ФІКСОВАНОЮ моделлю.
            - funding_skew / liquidity — rolling, каузальні.

        Args (2A):
            funding: series funding-ставки (опційно) → funding_skew z-score.
            volume: series обсягу (опційно) → liquidity label.
            refit_every: якщо >0, walk-forward refit HMM кожні `refit_every` барів
                на каузальному вікні [t-fit_window, t) (2A). 0 = single early fit
                (дефолт, як раніше). Рефіт дорогий; вмикати для довгих рядів.
        """
        if not self._fitted and refit_every <= 0:
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
        # Структура зі СТАРШОГО ТФ (опційно): лише закриті htf-вікна ≤ t.
        if self.htf_structure:
            state_df["structure"] = htf_market_structure(close, htf=self.htf_structure)
            state_df["label"] = state_df["structure"].astype(str) + "|" + state_df["vol"].astype(str)
        # Гістерезис структури (опційний) — після формування label/vol:
        # label перераховується під згладжену structure.
        if self.min_dwell_bars > 0:
            state_df["structure"] = apply_min_dwell(state_df["structure"], self.min_dwell_bars)
            state_df["label"] = state_df["structure"].astype(str) + "|" + state_df["vol"].astype(str)

        # HMM filtered proba (без lookahead)
        hmm_cols: dict[str, pd.Series] = {}
        if self._fitted and self._hmm is not None:
            ret = close.pct_change().fillna(0.0)
            vol = ret.rolling(20, min_periods=10).std().fillna(0.0)
            obs = pd.DataFrame({"ret": ret, "abs_ret": ret.abs(), "vol": vol}).iloc[20:]
            X = obs.values

            try:
                if refit_every > 0 and len(X) > self.hmm_fit_bars:
                    # Walk-forward refit (2A): кожні refit_every барів нова HMM на
                    # каузальному вікні [i-fit_window, i). filtered_proba per-segment.
                    post = self._filtered_proba_windowed(obs, X, refit_every)
                else:
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

        # Funding skew + liquidity (2A) — каузальні rolling-фічі.
        if funding is not None:
            state_df["funding_skew"] = _funding_skew_series(funding).reindex(close.index).fillna(0.0).values
        else:
            state_df["funding_skew"] = 0.0
        if volume is not None:
            state_df["liquidity"] = (
                _liquidity_label_from_volume(close, volume).reindex(close.index).fillna("normal").values
            )
        else:
            state_df["liquidity"] = "normal"

        # Заповнення відсутніх колонок
        for col in ["hmm_state", "confidence"] + list(hmm_cols.keys()):
            if col not in state_df.columns:
                state_df[col] = 0.0

        state_df["hmm_state"] = state_df["hmm_state"].ffill().fillna(-1).astype(int)
        state_df["confidence"] = state_df["confidence"].ffill().fillna(0.0)
        state_df = state_df.fillna(0.0)
        state_df["hmm_state"] = state_df["hmm_state"].astype(int)
        # joint_label (2A): єдина «правда» для роутингу.
        fsign = np.sign(state_df["funding_skew"].astype(float))
        ftag = np.where(fsign > 0.5, "p", np.where(fsign < -0.5, "n", "0"))
        state_df["joint_label"] = (
            state_df["label"].astype(str)
            + "|h"
            + state_df["hmm_state"].astype(str)
            + "|f"
            + ftag
            + "|"
            + state_df["liquidity"].astype(str)
        )

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
                    funding_skew=float(row.get("funding_skew", 0.0)),
                    liquidity=str(row.get("liquidity", "normal")),
                )
            )
        return states

    # ── Walk-forward refit (2A) ──────────────────────────────────────────────

    def refit(self, close: pd.Series, *, start: int = 0, end: int | None = None) -> RegimeDetector:
        """Перенавчання HMM на каузальному вікні close[start:end] (2A).

        На відміну від fit() (перші hmm_fit_bars), refit дозволяє довільне
        каузальне вікно для walk-forward. Мутує self._hmm. Використовується
        detect(..., refit_every=N) під капотом; можна викликати вручну для
        інкрементального live-refit.
        """
        ret = close.pct_change().fillna(0.0)
        vol = ret.rolling(20, min_periods=10).std().fillna(0.0)
        obs = pd.DataFrame({"ret": ret, "abs_ret": ret.abs(), "vol": vol}).iloc[20:]
        end_i = len(obs) if end is None else min(end, len(obs))
        window = obs.iloc[start:end_i]
        if len(window) < self.n_hmm_states * 10:
            logger.warning("RegimeDetector.refit: замало даних (%d). HMM не перенавчено.", len(window))
            return self
        try:
            self._hmm = GaussianHMM(n_states=self.n_hmm_states, seed=self.hmm_seed).fit(window.values)
            self._fitted = True
        except ValueError as exc:
            logger.warning("RegimeDetector.refit: HMM не навчився: %s", exc)
        return self

    def _filtered_proba_windowed(self, obs: pd.DataFrame, X: np.ndarray, refit_every: int) -> np.ndarray:
        """Walk-forward filtered_proba з каузальним refit (2A, без lookahead).

        Серію розбиваємо на сегменти по refit_every барів. Для сегмента [i, i+L):
        HMM навчається на obs[max(0, i-fit_window):i] (строго до сегмента),
        filtered_proba рахується лише на сегменті. Перший сегмент використовує
        модель з fit() (ранній fit). Конкатенуємо → (T, K).
        """
        T, K = X.shape
        fit_window = self.hmm_fit_bars
        out = np.zeros((T, K), dtype=float)
        # Перший сегмент: вже навчена модель (fit на перших hmm_fit_bars).
        seg_end = min(refit_every, T)
        if self._hmm is not None:
            try:
                out[:seg_end] = self._hmm.filtered_proba(X[:seg_end])
            except Exception as exc:  # noqa: BLE001
                logger.warning("RegimeDetector WF: перший сегмент помилка: %s", exc)
        # Подальші сегменти з каузальним refit.
        i = seg_end
        while i < T:
            j = min(i + refit_every, T)
            train_start = max(0, i - fit_window)
            train = X[train_start:i]
            if len(train) >= self.n_hmm_states * 10:
                try:
                    seg_hmm = GaussianHMM(n_states=self.n_hmm_states, seed=self.hmm_seed).fit(train)
                    out[i:j] = seg_hmm.filtered_proba(X[i:j])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("RegimeDetector WF: сегмент %d помилка: %s", i, exc)
                    if self._hmm is not None:
                        try:
                            out[i:j] = self._hmm.filtered_proba(X[i:j])
                        except Exception:
                            pass
            elif self._hmm is not None:
                try:
                    out[i:j] = self._hmm.filtered_proba(X[i:j])
                except Exception:
                    pass
            i = j
        return out

    # ── Online (live) ────────────────────────────────────────────────────────

    def step(
        self,
        new_close: float,
        funding_rate: float | None = None,
        volume: float | None = None,
    ) -> RegimeState:
        """Інкрементальне оновлення режиму для live (один бар).

        Буфер накопичує закриті ціни. Коли накопичено достатньо даних —
        автоматично навчає HMM (один раз). Повертає RegimeState поточного бару.

        Args (2A):
            funding_rate: поточна funding-ставка (опційно) → funding_skew z-score
                рахується по буферу funding-ставок.
            volume: обсяг поточного бару (опційно) → liquidity label.
        """
        self._close_buf.append(float(new_close))
        if funding_rate is not None:
            self._funding_buf.append(float(funding_rate))
        if volume is not None:
            self._volume_buf.append(float(volume))

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
                funding_skew=self._current_funding_skew(),
                liquidity=self._current_liquidity(),
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
            funding_skew=self._current_funding_skew(),
            liquidity=self._current_liquidity(),
        )
        self._last_state = state
        return state

    # ── Funding/liquidity live helpers (2A) ─────────────────────────────────
    def _current_funding_skew(self) -> float:
        if len(self._funding_buf) < 20:
            return 0.0
        f = pd.Series(self._funding_buf, dtype=float)
        z = _funding_skew_series(f, window=min(720, max(20, len(f) // 2)))
        return float(z.iloc[-1]) if len(z) else 0.0

    def _current_liquidity(self) -> str:
        if len(self._volume_buf) < 40 or len(self._close_buf) < 40:
            return "normal"
        c = pd.Series(self._close_buf[-len(self._volume_buf) :], dtype=float)
        v = pd.Series(self._volume_buf, dtype=float)
        lab = _liquidity_label_from_volume(c, v, window=min(120, max(20, len(v) // 2)))
        return str(lab.iloc[-1]) if len(lab) else "normal"

    @property
    def last_state(self) -> RegimeState | None:
        """Останній обчислений стан (None до першого step())."""
        return self._last_state

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def hmm_calm_state(self) -> int | None:
        """Індекс HMM-стану з найменшою волатильністю («найспокійніший» режим).

        Спирається на дизайн спостережень [ret, |ret|, vol]: стан із мінімальним
        covars_[:, 2] (дисперсія vol) — спокійний. None, якщо HMM не навчений.
        Використовується live-гейтом `LiveTrader.hmm_blocked` та мета-шаром для
        єдиної семантики «calm regime» у всьому проєкті.
        """
        if self._hmm is None or self._hmm.covars_ is None:
            return None
        return int(np.argmin(self._hmm.covars_[:, 2]))

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
                        "min_dwell_bars": self.min_dwell_bars,
                        "htf_structure": self.htf_structure,
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
