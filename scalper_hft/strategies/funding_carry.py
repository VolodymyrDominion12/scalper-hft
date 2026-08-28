"""Funding-carry стратегія: збір фандінгу на перекосах (дослідження: funding
harvest — найдоступніший retail-edge на перп-ф'ючерсах).

Логіка (книга, гл. 3 — mean reversion на ринковій структурі):
    - фандінг = плата за перекіс позицій: funding >> 0 → натовп у лонгах
      (лонги платять шортам); funding << 0 → натовп у шортах;
    - вхід у шорт, коли попередня ставка > upper_threshold (збираємо фандінг
      І очікуємо повернення перекосу);
    - вхід у лонг, коли попередня ставка < lower_threshold;
    - вихід: ставка повернулась у нейтральну зону (|fr| < exit_threshold).

Без lookahead: рішення приймається за ставкою, що вже була опублікована
(попередній період); позиція тримається між ставками (зазвичай 8h на Binance).
Грошовий потік фандінгу додає рушій бектесту (лонг платить позитивний фандінг).
"""

from __future__ import annotations

import pandas as pd

from scalper_hft.strategies.base import Strategy


class FundingCarryScalper(Strategy):
    name = "funding_carry"
    needs_funding = True

    param_space = {
        "upper_threshold": (0.00004, 0.0002, 0.00001),
        "lower_threshold": (-0.0002, -0.00004, 0.00001),
        "exit_threshold": (0.000005, 0.00005, 0.000005),
    }

    def __init__(
        self,
        upper_threshold: float = 0.00006,
        lower_threshold: float = -0.00006,
        exit_threshold: float = 0.00002,
    ) -> None:
        super().__init__(
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            exit_threshold=exit_threshold,
        )

    def generate_signals(self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None) -> pd.Series:
        """Сигнали на основі фандінгу, вирівняного на свічковий індекс.

        funding: DataFrame з колонкою 'fundingRate' (індекс — час ставки).
        """
        if funding is None or funding.empty:
            return pd.Series(0, index=df.index, dtype=int)

        upper = float(self.get("upper_threshold", 0.00006))
        lower = float(self.get("lower_threshold", -0.00006))
        exit_th = float(self.get("exit_threshold", 0.00002))

        fr = funding["fundingRate"].sort_index()
        # рішення за ПОПЕРЕДНЬОЮ ставкою (вже опублікованою — без lookahead)
        prev_fr = fr.shift(1).fillna(0.0)

        target = pd.Series(0.0, index=fr.index)
        target[prev_fr > upper] = -1.0  # шорт: збираємо позитивний фандінг
        target[prev_fr < lower] = 1.0  # лонг: збираємо негативний фандінг
        target[prev_fr.abs() < exit_th] = 0.0  # нейтральна зона

        # позиція діє від ставки t до наступної ставки (ffill на бари)
        sig = target.reindex(df.index, method="ffill").fillna(0.0)
        return sig.astype(int)
