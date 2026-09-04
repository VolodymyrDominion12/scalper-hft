"""Delta-neutral funding arbitrage (перп + спот).

Концепція (дослідження: funding harvest — найдоступніший retail-edge):
    - фандінг на перп-ф'ючерсах = плата за перекіс позицій;
    - якщо funding >> 0 (натовп у лонгах): ШОРТ перпа + ЛОНГ споту —
      ціновий ризик хеджується (delta-neutral), а позитивний фандінг збирається;
    - якщо funding << 0: ЛОНГ перпа + ШОРТ споту (дзеркально);
    - вихід: funding повернувся в нейтральну зону.

Прибуток ≈ зібраний фандінг + basis-зсув (невеликий) − комісії двох ніг.
Без цінового ризику напрямку — на відміну від funding_carry (односторонній).

⚠ Обмеження: Binance spot не дозволяє retail-шорт у звичайному акаунті —
плече -1 (long perp/short spot) потребує margin-рахунку. У бектесті обидва
напрямки моделюються; для live — починати з плеча +1 (short perp/long spot).
"""

from __future__ import annotations

import pandas as pd

from scalper_hft.strategies.base import Strategy


class FundingArb(Strategy):
    name = "funding_arb"
    family = "carry"
    preferred_regimes = frozenset()
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

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        """Carry-сигнал пари: +1 = шорт перп/лонг спот (збір позитивного фандінгу),
        −1 = лонг перп/шорт спот. Рішення за ПОПЕРЕДНЬОЮ ставкою (без lookahead)."""
        if funding is None or funding.empty:
            return pd.Series(0, index=df.index, dtype=int)

        upper = float(self.get("upper_threshold", 0.00006))
        lower = float(self.get("lower_threshold", -0.00006))
        exit_th = float(self.get("exit_threshold", 0.00002))

        fr = funding["fundingRate"].sort_index()
        prev_fr = fr.shift(1).fillna(0.0)

        target = pd.Series(0.0, index=fr.index)
        target[prev_fr > upper] = 1.0  # carry +1: шорт перп (збір фандінгу)
        target[prev_fr < lower] = -1.0  # carry -1: лонг перп
        target[prev_fr.abs() < exit_th] = 0.0

        sig = target.reindex(df.index, method="ffill").fillna(0.0)
        return sig.astype(int)
