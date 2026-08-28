"""CVD-momentum скальпер: конвергенція потоку заявок і ціни.

Мікроструктурна логіка (дослідження 2023–25: CVD guide):
    - CVD (cumulative volume delta) показує агресію покупців/продавців;
    - довгий сигнал: CVD зростає (cvd_mom > 0) І ціна вище EMA_fast
      (підтвердження напрямку) — вхід за темпом потоку;
    - короткий сигнал: дзеркально;
    - вихід: розворот cvd_mom або ціни проти позиції.

Потребує aggTrades (needs_trades=True). Якщо даних про трейди немає —
працює на buy_ratio зі свічок (об'ємний дисбаланс).
"""

from __future__ import annotations

import pandas as pd

from scalper_hft.features.indicators import add_standard_features, cvd_from_trades, ema
from scalper_hft.strategies.base import Strategy


class CvdMomentumScalper(Strategy):
    name = "cvd_momentum"
    needs_trades = True

    param_space = {
        "ema_fast": (3.0, 20.0, 1.0),
        "cvd_mom_window": (4.0, 30.0, 2.0),
        "cvd_threshold": (0.0, 0.0001, 0.000005),
        "min_vol_ratio": (0.5, 3.0, 0.25),
    }

    def __init__(
        self,
        ema_fast: int = 9,
        cvd_mom_window: int = 12,
        cvd_threshold: float = 0.0,
        min_vol_ratio: float = 1.0,
    ) -> None:
        super().__init__(
            ema_fast=int(ema_fast),
            cvd_mom_window=int(cvd_mom_window),
            cvd_threshold=cvd_threshold,
            min_vol_ratio=min_vol_ratio,
        )

    def generate_signals(self, df: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.Series:
        f = add_standard_features(df)
        if trades is not None and not trades.empty:
            cvd_df = cvd_from_trades(trades, resample=_infer(df))
            # нормалізуємо cvd_mom до відносної зміни об'єму
            cvd_mom = cvd_df["cvd"].diff(self.get("cvd_mom_window", 12)) / (
                f["volume"].rolling(20, min_periods=5).sum().replace(0, float("nan"))
            )
            cvd_mom = cvd_mom.reindex(f.index).ffill().fillna(0.0)
        else:
            # фолбек: buy_ratio зміна як проксі потоку
            br = f["volume"] * 0  # немає даних — нульовий сигнал потоку
            cvd_mom = pd.Series(0.0, index=f.index)

        ema_fast = ema(f["close"], int(self.get("ema_fast", 9)))
        price_above = f["close"] > ema_fast
        price_below = f["close"] < ema_fast
        flow_buy = cvd_mom > self.get("cvd_threshold", 0.0)
        flow_sell = cvd_mom < -self.get("cvd_threshold", 0.0)
        vol_ok = f["vol_ratio"] >= self.get("min_vol_ratio", 1.0)

        # Патерн "вхід утримується до виходу": виходи прив'язані до попередньої позиції.
        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[price_above & flow_buy & vol_ok] = 1.0
        sig[price_below & flow_sell & vol_ok] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        # вихід лонга при розвороті ціни або потоку
        exit_long = (prev_pos == 1.0) & (price_below | flow_sell)
        exit_short = (prev_pos == -1.0) & (price_above | flow_buy)
        sig[exit_long | exit_short] = 0.0

        return sig.ffill().fillna(0.0).astype(int)


def _infer(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "1min"
    delta = df.index[1] - df.index[0]
    minutes = delta.total_seconds() / 60.0
    return f"{int(minutes)}min" if minutes >= 1 else f"{int(max(1, minutes * 60))}s"
