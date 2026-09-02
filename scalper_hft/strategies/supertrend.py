"""Supertrend (ATR-based trailing trend) — порт з trade-bot-main.

Оригінал: trade-bot-main/strategies/supertrend_strategy.py (+ Indicators.supertrend).

Логіка (портовано у scalper-конвенції, Series позицій):
    - лінія Supertrend рахується на закритті бару t (стан-машина за
      `final_upper/final_lower` — класична формула ATR-bands);
    - напрямок (+1 бик / −1 ведмідь) фліпається, коли close перетинає стрічку;
    - вхід у лонг — лише на фліпі вгору **з підтвердженням об'єму**
      (volume > volume_factor × SMA20), як в оригіналі;
    - вхід у шорт — на фліпі вниз (лише якщо `allow_short=True`; за
      замовчуванням −1 = вихід з лонга, як у спотовому оригіналі);
    - SL/TP — від ціни входу: SL = min(лінія ST, close − ATR×sl),
      TP = close + ATR×tp (clip 0.8×/1.8× ціни, як в оригіналі);
      вихід за рівнем — на дотику бару (low/high).

Увага: стратегія stateful, тому використовує python-цикл (як і оригінал).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.features.indicators import atr, sma
from scalper_hft.strategies.base import Strategy


class SupertrendStrategy(Strategy):
    name = "supertrend"

    param_space = {
        "atr_period": (7.0, 14.0, 1.0),
        "atr_mult": (2.0, 4.0, 0.25),
        "atr_mult_sl": (1.5, 3.0, 0.25),
        "atr_mult_tp": (3.0, 7.0, 0.5),
        "volume_factor": (0.8, 1.6, 0.1),
    }

    def __init__(
        self,
        atr_period: int = 10,
        atr_mult: float = 3.0,
        atr_mult_sl: float = 2.0,
        atr_mult_tp: float = 4.5,
        volume_factor: float = 1.0,
        allow_short: bool = False,
    ) -> None:
        super().__init__(
            atr_period=int(atr_period),
            atr_mult=float(atr_mult),
            atr_mult_sl=float(atr_mult_sl),
            atr_mult_tp=float(atr_mult_tp),
            volume_factor=float(volume_factor),
            allow_short=bool(allow_short),
        )

    def _supertrend_lines(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Лінія Supertrend та напрямок (+1/−1) на закритті кожного бару.

        ATR рахується локально через `ewm(span=…)` (як в оригіналі): без
        min_periods значення є з першого бару, тому рекурсивні стрічки
        Supertrend не «застрягають» на NaN під час прогріву.
        """
        period = int(self.get("atr_period", 10))
        mult = float(self.get("atr_mult", 3.0))
        high_s = df["high"]
        low_s = df["low"]
        close_s = df["close"]
        tr = pd.concat(
            [high_s - low_s, (high_s - close_s.shift()).abs(), (low_s - close_s.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr_v = tr.ewm(span=period, adjust=False).mean().to_numpy(dtype=float)
        hl2 = ((high_s + low_s) / 2.0).to_numpy(dtype=float)
        close = close_s.to_numpy(dtype=float)

        basic_upper = hl2 + mult * atr_v
        basic_lower = hl2 - mult * atr_v

        n = len(df)
        final_upper = np.empty(n)
        final_lower = np.empty(n)
        direction = np.ones(n, dtype=np.int8)
        st = np.empty(n)

        final_upper[0] = basic_upper[0]
        final_lower[0] = basic_lower[0]
        direction[0] = 1
        st[0] = final_lower[0]

        for i in range(1, n):
            bu, bl = basic_upper[i], basic_lower[i]
            if np.isnan(bu) or np.isnan(bl):
                final_upper[i] = final_upper[i - 1]
                final_lower[i] = final_lower[i - 1]
                direction[i] = direction[i - 1]
                st[i] = st[i - 1]
                continue
            # Стрічки «затягуються» лише в один бік (класика Supertrend).
            if bu < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = bu
            else:
                final_upper[i] = final_upper[i - 1]
            if bl > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = bl
            else:
                final_lower[i] = final_lower[i - 1]

            if direction[i - 1] == 1:
                direction[i] = -1 if close[i] < final_lower[i] else 1
            else:
                direction[i] = 1 if close[i] > final_upper[i] else -1
            st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

        return st, direction

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        st, direction = self._supertrend_lines(df)
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        n = len(df)

        vol = df["volume"].to_numpy(dtype=float)
        vol_sma = sma(df["volume"], window=20).to_numpy(dtype=float)
        volume_factor = float(self.get("volume_factor", 1.0))
        vol_ok = np.zeros(n, dtype=bool)
        valid = np.isfinite(vol_sma) & (vol_sma > 0)
        vol_ok[valid] = vol[valid] > volume_factor * vol_sma[valid]

        allow_short = bool(self.get("allow_short", False))
        sl_mult = float(self.get("atr_mult_sl", 2.0))
        tp_mult = float(self.get("atr_mult_tp", 4.5))
        atr14 = atr(df, period=14).to_numpy(dtype=float)

        pos = np.zeros(n, dtype=np.int8)
        sl_price = np.full(n, np.nan)
        tp_price = np.full(n, np.nan)
        cur: int = 0
        cur_sl = cur_tp = float("nan")

        for i in range(1, n):
            d = int(direction[i])
            prev_d = int(direction[i - 1])
            flip_up = d == 1 and prev_d == -1
            flip_down = d == -1 and prev_d == 1
            price = float(close[i])

            if cur == 0:
                if flip_up and vol_ok[i]:
                    cur = 1
                    stop = min(float(st[i]), price - sl_mult * atr14[i])
                    cur_sl = max(stop, price * 0.80)
                    cur_tp = min(price + tp_mult * atr14[i], price * 1.80)
                elif allow_short and flip_down:
                    cur = -1
                    cur_sl = price + sl_mult * atr14[i]
                    cur_tp = price - tp_mult * atr14[i]
            elif cur == 1:
                if flip_down:
                    cur = -1 if allow_short else 0
                    if cur == -1:
                        cur_sl = price + sl_mult * atr14[i]
                        cur_tp = price - tp_mult * atr14[i]
                elif low[i] <= cur_sl or high[i] >= cur_tp:
                    cur = 0
            else:  # cur == -1
                if flip_up:
                    # Об'ємний фільтр — лише для лонгів (як в оригіналі).
                    cur = 1 if vol_ok[i] else 0
                    if cur == 1:
                        stop = min(float(st[i]), price - sl_mult * atr14[i])
                        cur_sl = max(stop, price * 0.80)
                        cur_tp = min(price + tp_mult * atr14[i], price * 1.80)
                elif high[i] >= cur_sl or low[i] <= cur_tp:
                    cur = 0

            pos[i] = cur
            sl_price[i] = cur_sl if cur != 0 else float("nan")
            tp_price[i] = cur_tp if cur != 0 else float("nan")

        return pd.Series(pos, index=df.index, dtype=int)
