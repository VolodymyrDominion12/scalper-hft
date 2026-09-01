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
    ) -> None:
        super().__init__(
            entry_z=entry_z,
            exit_z=exit_z,
            lookback=int(lookback),
            breakeven_gate=bool(breakeven_gate),
            hmm_vol_gate=bool(hmm_vol_gate),
        )

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        """Сигнал пари: +1 = шорт leg1/лонг leg2 (спред високий), −1 = дзеркально.

        df: DataFrame з колонками leg1/leg2 (ціни закриття перпів).
        Сигнал на закритті t → виконання t+1 (рушій робить shift).
        """
        if "leg1" not in df.columns or "leg2" not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)

        ratio = np.log(df["leg1"] / df["leg2"])
        lookback = int(self.get("lookback", 480))
        entry_z = float(self.get("entry_z", 2.0))
        exit_z = float(self.get("exit_z", 0.3))
        use_breakeven_gate = bool(self.get("breakeven_gate", False))
        use_hmm_vol_gate = bool(self.get("hmm_vol_gate", False))

        mean = ratio.rolling(lookback, min_periods=lookback // 2).mean()
        std = ratio.rolling(lookback, min_periods=lookback // 2).std(ddof=0).replace(0, np.nan)
        z = ((ratio - mean) / std).fillna(0.0)

        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[z > entry_z] = 1.0
        sig[z < -entry_z] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        exit_any = (prev_pos != 0.0) & (z.abs() < exit_z)
        sig[exit_any] = 0.0

        sig = sig.ffill().fillna(0.0).astype(int)

        # ── Breakeven gate: відключаємо сигнали де ATR спреду < 2-leg round-trip ──
        if use_breakeven_gate:
            sig = self._apply_breakeven_gate(sig, ratio, df)

        # ── HMM-фільтр: відключаємо входи у стані максимальної волатильності ──
        if use_hmm_vol_gate:
            sig = self._apply_hmm_vol_gate(sig, df)

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

