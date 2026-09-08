"""Статистичний арбітраж пар (perps BTC/ETH/SOL).

Ідея: висококорельовані пари (BTC/ETH, BTC/SOL, ETH/SOL) мають середньо-
реверсійний log-ratio. Коли спред відхиляється від норми — шортуємо
"переоцінену" ногу і лонгуємо "недооцінену" (delta-neutral пара на перпах).

Переваги проти funding/basis:
    - обидві ноги на USDT-M перпах (без spot-short обмеження);
    - спред-рухи більші за basis (можуть покрити 2-leg комісії);
    - збір фандінгу обох ніг частково компенсує витрати.

⚠ Ризики: спред може не ревертнутись (тренд у відносній силі — напр. ETH
випереджає BTC у ралі); обидві ноги мають фандінг — потрібен чистий облік.

Нові параметри (покращення результативності):
    breakeven_gate  : bool = False — відключає сигнали де ATR < round-trip витрат.
                      Потребує передачі cost_model ззовні або дефолтного CostModel.
    hmm_vol_gate    : bool = False — відключає входи у HMM-стані max-волатильності
                      (state 2 при n_states=3). М'якше за hmm_reversion: лише
                      найгірший стан блокується.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.strategies.base import Strategy


class PairsArb(Strategy):
    name = "pairs_arb"
    family = "relative_value"
    preferred_regimes = frozenset()

    param_space = {
        "entry_z": (1.5, 4.0, 0.25),
        "exit_z": (0.0, 1.0, 0.1),
        "lookback": (60.0, 1440.0, 60.0),
    }

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.3,
        lookback: int = 480,
        breakeven_gate: bool = False,
        hmm_vol_gate: bool = False,
        use_kalman: bool = False,
        kalman_q: float = 1e-5,
        kalman_r: float = 1e-3,
        dynamic_half_life: bool = False,
        regime_scale: bool = False,
        regime_scale_factor: float = 0.5,
    ) -> None:
        super().__init__(
            entry_z=entry_z,
            exit_z=exit_z,
            lookback=int(lookback),
            breakeven_gate=bool(breakeven_gate),
            hmm_vol_gate=bool(hmm_vol_gate),
            use_kalman=bool(use_kalman),
            kalman_q=float(kalman_q),
            kalman_r=float(kalman_r),
            dynamic_half_life=bool(dynamic_half_life),
            regime_scale=bool(regime_scale),
            regime_scale_factor=float(regime_scale_factor),
        )
        self.betas: pd.Series | None = None

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        """Сигнал пари: +1 = шорт leg1/лонг leg2 (спред високий), −1 = дзеркально.

        df: DataFrame з колонками leg1/leg2 (ціни закриття перпів).
        Сигнал на закритті t → виконання t+1 (рушій робить shift).
        """
        # Скидаємо betas на початку: інакше при early-return сюди потрапляє
        # застаріле значення з ПОПЕРЕДНЬОГО виклику (stale-bleed у pairs.py).
        self.betas = None
        if "leg1" not in df.columns or "leg2" not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)

        use_kalman = bool(self.get("use_kalman", False))
        dynamic_half_life = bool(self.get("dynamic_half_life", False))

        if use_kalman:
            from scalper_hft.features.signal_processing import dynamic_hedge_ratio

            kq = float(self.get("kalman_q", 1e-5))
            kr = float(self.get("kalman_r", 1e-3))
            kf_res = dynamic_hedge_ratio(np.log(df["leg1"]), np.log(df["leg2"]), q_beta=kq, q_alpha=kq, r=kr)
            spread_series = kf_res["spread"]
            self.betas = kf_res["beta"]
        else:
            spread_series = np.log(df["leg1"] / df["leg2"])
            self.betas = pd.Series(1.0, index=df.index)

        lookback = int(self.get("lookback", 480))
        if dynamic_half_life:
            from scalper_hft.features.signal_processing import estimate_half_life

            hl = estimate_half_life(spread_series.tail(min(lookback, len(spread_series))))
            if np.isfinite(hl) and hl > 5:
                lookback = int(np.clip(hl * 2.0, 30.0, 1440.0))

        entry_z = float(self.get("entry_z", 2.0))
        exit_z = float(self.get("exit_z", 0.3))
        use_breakeven_gate = bool(self.get("breakeven_gate", False))
        use_hmm_vol_gate = bool(self.get("hmm_vol_gate", False))

        mean = spread_series.rolling(lookback, min_periods=lookback // 2).mean()
        std = spread_series.rolling(lookback, min_periods=lookback // 2).std(ddof=0).replace(0, np.nan)
        z = ((spread_series - mean) / std).fillna(0.0)

        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[z > entry_z] = 1.0
        sig[z < -entry_z] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        exit_any = (prev_pos != 0.0) & (z.abs() < exit_z)
        sig[exit_any] = 0.0

        sig = sig.ffill().fillna(0.0).astype(int)

        # ── Breakeven gate: відключаємо сигнали де ATR спреду < 2-leg round-trip ──
        if use_breakeven_gate:
            sig = self._apply_breakeven_gate(sig, spread_series, df)

        # ── HMM-фільтр: відключаємо входи у стані максимальної волатильності ──
        if use_hmm_vol_gate:
            sig = self._apply_hmm_vol_gate(sig, df)

        # ── Regime-scale: масштабуємо силу сигналу за режимом leg2 ──
        if bool(self.get("regime_scale", False)):
            factor = float(self.get("regime_scale_factor", 0.5))
            sig = self._apply_regime_scale(sig, df, factor=factor)

        return sig

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_breakeven_gate(sig: pd.Series, ratio: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Блокує сигнали де ATR спреду (ratio) < maker round-trip витрат.

        ratio = log(leg1/leg2) — вже безрозмірна величина (частка ціни),
        тому ATR спреду напряму порівнюється з round-trip costs у %.
        Для 2-leg позиції: threshold = 2 × maker round-trip ≈ 0.08%.
        """
        from scalper_hft.backtest.execution import CostModel

        cost = CostModel()
        # ATR спреду: середнє |Δratio| за 14 барів
        ratio_atr = ratio.diff().abs().rolling(14, min_periods=7).mean()
        # ratio вже у fractional units (log-scale ≈ fractional) — не ділимо на ціну
        threshold = cost.round_trip_maker() * 2.0
        gate = ratio_atr >= threshold
        # зберігаємо позиції що вже відкриті (не перекриваємо виходи)
        holding = sig.shift(1).fillna(0.0) != 0.0
        return sig.where(gate.reindex(sig.index, fill_value=False) | holding, other=0)

    @staticmethod
    def _apply_hmm_vol_gate(sig: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Блокує нові входи у HMM-стані з максимальною волатильністю.

        Відмінно від hmm_reversion: блокує лише state 2 (найгірший стан
        при n_states=3), тому ≈60-70% сигналів зберігаються.
        causal=True — без lookahead.
        """
        try:
            from scalper_hft.features.hmm_regime import hmm_regime_features

            close = df.get("leg1", df.get("close", None))
            if close is None or len(close) < 100:
                return sig
            hmm = hmm_regime_features(close, n_states=3, causal=True)
            # state 2 = найвища волатильність (сортування за середнім ret → state 2 ≠ завжди vol!)
            # використовуємо hmm_p2 (posterior P(state 2)) як міру небезпеки
            p_high_vol = hmm.get("hmm_p2", pd.Series(0.0, index=hmm.index))
            p_high_vol_aligned = p_high_vol.reindex(sig.index, method="ffill").fillna(0.0)
            # блокуємо лише якщо P(state 2) > 0.6 (тверда впевненість у небезпечному стані)
            block = p_high_vol_aligned > 0.6
            # блокуємо лише нові входи; виходи/утримання позиції — не блокуємо
            holding = sig.shift(1).fillna(0.0) != 0.0
            return sig.where(~block | holding, other=0)
        except Exception:  # noqa: BLE001
            return sig

    @staticmethod
    def _apply_regime_scale(sig: pd.Series, df: pd.DataFrame, factor: float = 0.5) -> pd.Series:
        """Масштабувати силу сигналу за режимом leg2 (ринковий годинник).

        leg2 — зазвичай BTC (валюта котирування пари). У спокійному range/normal
        mean-reversion спреду працює краще; у high-vol або тренді BTC відносна
        сила може трендити (break co-integration) → зменшуємо експозицію до factor.

        Каузально: regime обчислюється на close ≤ t (named_market_state —
        EMA/vol-percentile без lookahead). Розмір фіксується на моменті входу
        (ffill entry-scale під час угоди) — без resize-churn між барами.
        Виходи (sig==0) та утримання напрямку зберігають повний розмір, як і
        інші гейти (hmm_vol_gate/breakeven_gate): лише нові входи масштабуються.
        """
        from scalper_hft.features.regimes import named_market_state

        leg2 = df.get("leg2")
        if leg2 is None or len(leg2) < 100:
            return sig.astype(float)
        state = named_market_state(leg2)
        vol = state["vol"].reindex(sig.index, method="ffill").fillna("normal")
        structure = state["structure"].reindex(sig.index, method="ffill").fillna("range")
        # scale: 1.0 у сприятливому (range/normal/low), factor у несприятливому
        scale = pd.Series(1.0, index=sig.index)
        scale = scale.where(~(vol == "high"), factor)
        scale = scale.where(~structure.isin(["trend_up", "trend_down"]), factor)

        sig_f = sig.astype(float)
        # новий вхід: sig != 0 після sig == 0
        new_entry = (sig_f != 0.0) & (sig_f.shift(1).fillna(0.0) == 0.0)
        entry_scale = pd.Series(np.nan, index=sig.index, dtype=float)
        entry_scale.loc[new_entry.values] = scale.loc[new_entry.values].values
        # lock entry-time scale на час угоди (ffill); поза позицією — байдуже (sig==0)
        entry_scale = entry_scale.ffill().fillna(1.0)
        return (sig_f * entry_scale).clip(-1.0, 1.0)
