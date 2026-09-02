"""Stochastic RSI mean-reversion — порт з trade-bot-main.

Оригінал: trade-bot-main/strategies/stoch_rsi_strategy.py (+ Indicators.stoch_rsi).

Логіка (портовано у scalper-конвенції, Series позицій):
    - %K/%D Stochastic RSI (RSI нормалізований у [0..100] за власний min/max);
    - BUY:  %K перетинає %D вгору, виходячи із зони перепроданості
            (%K(t−1) < oversold), об'єм вищий за volume_factor×SMA20,
            close не нижче ema50×0.97 (фільтр «не ловимо падаючий ніж»);
    - SELL (вихід з лонга): %K перетинає %D вниз в зоні перекупленості
      АБО %K-cross вниз при RSI > 70;
    - SL/TP — ATR від ціни входу (clip 0.8×/1.6×, як в оригіналі); вихід
      за рівнем — на дотику бару (low/high);
    - `allow_short=True` робить «SELL» шортом (дзеркальні рівні), інакше —
      це просто закриття лонга (спотова поведінка оригіналу).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.features.indicators import atr, ema, rsi, sma
from scalper_hft.strategies.base import Strategy


class StochRsiStrategy(Strategy):
    name = "stoch_rsi"

    param_space = {
        "rsi_period": (10.0, 21.0, 1.0),
        "smooth_k": (2.0, 5.0, 1.0),
        "smooth_d": (2.0, 5.0, 1.0),
        "oversold": (15.0, 30.0, 2.5),
        "overbought": (70.0, 85.0, 2.5),
        "atr_mult_sl": (1.5, 3.0, 0.25),
        "atr_mult_tp": (2.5, 5.0, 0.5),
        "volume_factor": (0.8, 1.5, 0.1),
    }

    def __init__(
        self,
        rsi_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        atr_mult_sl: float = 2.0,
        atr_mult_tp: float = 3.5,
        volume_factor: float = 1.0,
        allow_short: bool = False,
    ) -> None:
        super().__init__(
            rsi_period=int(rsi_period),
            smooth_k=int(smooth_k),
            smooth_d=int(smooth_d),
            oversold=float(oversold),
            overbought=float(overbought),
            atr_mult_sl=float(atr_mult_sl),
            atr_mult_tp=float(atr_mult_tp),
            volume_factor=float(volume_factor),
            allow_short=bool(allow_short),
        )

    @staticmethod
    def _stoch_rsi(close: pd.Series, period: int, smooth_k: int, smooth_d: int) -> tuple[pd.Series, pd.Series]:
        rsi_v = rsi(close, period)
        rsi_min = rsi_v.rolling(period).min()
        rsi_max = rsi_v.rolling(period).max()
        stoch = (rsi_v - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100.0
        k = stoch.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        return k.fillna(50.0), d.fillna(50.0)

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        period = int(self.get("rsi_period", 14))
        smooth_k = int(self.get("smooth_k", 3))
        smooth_d = int(self.get("smooth_d", 3))
        oversold = float(self.get("oversold", 20.0))
        overbought = float(self.get("overbought", 80.0))
        sl_mult = float(self.get("atr_mult_sl", 2.0))
        tp_mult = float(self.get("atr_mult_tp", 3.5))
        volume_factor = float(self.get("volume_factor", 1.0))
        allow_short = bool(self.get("allow_short", False))

        close = df["close"]
        k, d = self._stoch_rsi(close, period, smooth_k, smooth_d)

        k_cross_up = (k.shift(1) <= d.shift(1)) & (k > d)
        from_oversold = k.shift(1) < oversold
        vol_ratio = df["volume"] / sma(df["volume"], window=20).replace(0.0, np.nan)
        volume_ok = vol_ratio > volume_factor
        not_crash = close > ema(close, span=50) * 0.97

        buy = (k_cross_up & from_oversold & volume_ok & not_crash).to_numpy(dtype=bool)

        k_cross_down = (k.shift(1) >= d.shift(1)) & (k < d)
        overbought_exit = (k > overbought) & k_cross_down
        rsi_exit = (k_cross_down & (rsi(close, period) > 70.0)).to_numpy(dtype=bool)
        sell = (overbought_exit | rsi_exit).to_numpy(dtype=bool)

        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        atr14 = atr(df, period=14).to_numpy(dtype=float)
        n = len(df)
        close_np = close.to_numpy(dtype=float)

        pos = np.zeros(n, dtype=np.int8)
        cur: int = 0
        cur_sl = cur_tp = float("nan")

        for i in range(n):
            price = float(close_np[i])
            if cur == 0:
                if buy[i]:
                    cur = 1
                    cur_sl = max(price - sl_mult * atr14[i], price * 0.80)
                    cur_tp = min(price + tp_mult * atr14[i], price * 1.60)
                elif allow_short and sell[i]:
                    cur = -1
                    cur_sl = price + sl_mult * atr14[i]
                    cur_tp = max(price - tp_mult * atr14[i], price * 0.20)
            elif cur == 1:
                if sell[i]:
                    cur = -1 if allow_short else 0
                    if cur == -1:
                        cur_sl = price + sl_mult * atr14[i]
                        cur_tp = max(price - tp_mult * atr14[i], price * 0.20)
                elif low[i] <= cur_sl or high[i] >= cur_tp:
                    cur = 0
            else:  # cur == -1
                if buy[i]:
                    cur = 1
                    cur_sl = max(price - sl_mult * atr14[i], price * 0.80)
                    cur_tp = min(price + tp_mult * atr14[i], price * 1.60)
                elif high[i] >= cur_sl or low[i] <= cur_tp:
                    cur = 0
            pos[i] = cur

        return pd.Series(pos, index=df.index, dtype=int)


__all__ = ["StochRsiStrategy"]
