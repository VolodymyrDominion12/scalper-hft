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
    ) -> None:
        super().__init__(
            entry_z=entry_z,
            exit_z=exit_z,
            lookback=int(lookback),
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

        mean = ratio.rolling(lookback, min_periods=lookback // 2).mean()
        std = ratio.rolling(lookback, min_periods=lookback // 2).std(ddof=0).replace(0, np.nan)
        z = ((ratio - mean) / std).fillna(0.0)

        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[z > entry_z] = 1.0
        sig[z < -entry_z] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        exit_any = (prev_pos != 0.0) & (z.abs() < exit_z)
        sig[exit_any] = 0.0

        return sig.ffill().fillna(0.0).astype(int)
