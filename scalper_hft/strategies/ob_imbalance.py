"""Order Book Imbalance скальпер: дисбаланс стакана як сигнал короткострокового напрямку.

Логіка (книга, гл. 14; дослідження: OB imbalance predict very short-term direction):
    - кожен бар/снапшот стакана дає imbalance = (bid_depth − ask_depth) / (bid + ask);
    - вхід у лонг, коли imbalance > buy_threshold (глибший бід = тиск покупців);
    - вхід у шорт, коли imbalance < -buy_threshold;
    - вихід: повернення imbalance до нейтральної зони (або контрсигнал).

Дані: bookTicker-снапшоти (best bid/ask + об'єми) або синтетичний imbalance
з aggTrades (buy_ratio). Для бектесту використовується подієвий рушій
(scalper_hft.backtest.event_engine).
"""

from __future__ import annotations

import pandas as pd

from scalper_hft.features.indicators import cvd_from_trades, zscore
from scalper_hft.strategies.base import Strategy


class ObImbalanceScalper(Strategy):
    name = "ob_imbalance"
    needs_trades = True

    param_space = {
        "buy_threshold": (0.05, 0.6, 0.05),
        "exit_threshold": (0.0, 0.3, 0.05),
        "lookback": (10.0, 200.0, 10.0),
    }

    def __init__(
        self,
        buy_threshold: float = 0.3,
        exit_threshold: float = 0.05,
        lookback: int = 50,
    ) -> None:
        super().__init__(
            buy_threshold=buy_threshold,
            exit_threshold=exit_threshold,
            lookback=int(lookback),
        )

    def generate_signals(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Працює з DataFrame, що містить колонку 'imbalance' (з bookTicker),
        або будує синтетичний imbalance з buy_ratio aggTrades."""
        if "imbalance" in df.columns:
            imb = df["imbalance"].fillna(0.0)
        elif trades is not None and not trades.empty:
            cvd = cvd_from_trades(trades, resample=_infer(df))
            # buy_ratio ∈ [0,1] → imbalance ∈ [−1,1]
            br = cvd["buy_ratio"].reindex(df.index).ffill().fillna(0.5)
            imb = 2.0 * br - 1.0
        else:
            imb = pd.Series(0.0, index=df.index)

        # z-нормалізація робить пороги стійкішими до рівня ліквідності
        imb_z = zscore(imb, int(self.get("lookback", 50)))

        buy_th = float(self.get("buy_threshold", 0.3))
        exit_th = float(self.get("exit_threshold", 0.05))

        # Патерн "вхід утримується до виходу": виходи прив'язані до попередньої позиції.
        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[imb_z > buy_th] = 1.0
        sig[imb_z < -buy_th] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        # вихід у нейтральній зоні (лише якщо в позиції)
        exit_any = (prev_pos != 0.0) & (imb_z.abs() < exit_th)
        sig[exit_any] = 0.0

        return sig.ffill().fillna(0.0).astype(int)


def _infer(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "1min"
    delta = df.index[1] - df.index[0]
    minutes = delta.total_seconds() / 60.0
    return f"{int(minutes)}min" if minutes >= 1 else f"{int(max(1, minutes * 60))}s"
