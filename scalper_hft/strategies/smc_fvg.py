"""SMC Fair Value Gap — порт з trade-bot-main.

Оригінал: trade-bot-main/strategies/smc_fvg_strategy.py (+ smc/liquidity.py::fvg).

Логіка (портовано у scalper-конвенції, Series позицій):
    - FVG-зона формується на закритті бару t (low[t] > high[t−2] — бичачий
      розрив; high[t] < low[t−2] — ведмежий). Доступна для торгівлі лише з
      наступних барів → без lookahead;
    - активна зона живе, поки її не закриє ціна (low < bottom для bull;
      high > top для bear) або не спливе `max_fvg_age` барів (як в оригіналі);
    - BUY:  перший дотик ціни до активної bull-зони (low у межах зони);
    - SHORT/exit: перший дотик до активної bear-зони (`allow_short=True` —
      шорт; інакше дотик = вихід з лонга, як у спотовому оригіналі);
    - SL/TP — ATR від ціни входу; вихід за рівнем — на дотику бару;
    - `volume_filter=True` — входити лише коли volume > SMA20 (як в оригіналі).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalper_hft.features.indicators import atr, sma
from scalper_hft.features.smc import fair_value_gaps
from scalper_hft.strategies.base import Strategy


class SmcFvgStrategy(Strategy):
    name = "smc_fvg"
    family = "flow"
    preferred_regimes = frozenset()

    param_space = {
        "atr_sl": (1.0, 3.0, 0.5),
        "atr_tp": (2.0, 6.0, 0.5),
        "max_fvg_age": (6.0, 48.0, 2.0),
    }

    def __init__(
        self,
        atr_sl: float = 2.0,
        atr_tp: float = 4.0,
        volume_filter: bool = False,
        allow_short: bool = False,
        max_fvg_age: int = 24,
    ) -> None:
        super().__init__(
            atr_sl=float(atr_sl),
            atr_tp=float(atr_tp),
            volume_filter=bool(volume_filter),
            allow_short=bool(allow_short),
            max_fvg_age=int(max_fvg_age),
        )

    @staticmethod
    def _active_fvg_levels(
        bull_fvg: np.ndarray,
        bear_fvg: np.ndarray,
        fvg_top: np.ndarray,
        fvg_bottom: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        max_age: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Каузальні активні FVG: зона активна з моменту формування,
        інвалідується закриттям ціною або за віком (`max_age` барів)."""
        n = len(bull_fvg)
        a_bull_top = np.full(n, np.nan)
        a_bull_bot = np.full(n, np.nan)
        a_bear_top = np.full(n, np.nan)
        a_bear_bot = np.full(n, np.nan)
        bt = bb = np.nan
        rt = rb = np.nan
        age_b = age_r = 0
        for i in range(n):
            if bull_fvg[i] == 1 and np.isfinite(fvg_top[i]) and np.isfinite(fvg_bottom[i]):
                bt, bb = max(fvg_top[i], fvg_bottom[i]), min(fvg_top[i], fvg_bottom[i])
                age_b = 0
            elif np.isfinite(bt):
                age_b += 1
                if low[i] < bb or age_b > max_age:
                    bt = bb = np.nan
                    age_b = 0
            a_bull_top[i], a_bull_bot[i] = bt, bb

            if bear_fvg[i] == 1 and np.isfinite(fvg_top[i]) and np.isfinite(fvg_bottom[i]):
                rt, rb = max(fvg_top[i], fvg_bottom[i]), min(fvg_top[i], fvg_bottom[i])
                age_r = 0
            elif np.isfinite(rt):
                age_r += 1
                if high[i] > rt or age_r > max_age:
                    rt = rb = np.nan
                    age_r = 0
            a_bear_top[i], a_bear_bot[i] = rt, rb
        return a_bull_top, a_bull_bot, a_bear_top, a_bear_bot

    def _run_state_machine(self, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """Спільний stateful-цикл: (сигнали, рівні SL/TP по барах)."""
        atr_sl = float(self.get("atr_sl", 2.0))
        atr_tp = float(self.get("atr_tp", 4.0))
        max_age = int(self.get("max_fvg_age", 24))
        allow_short = bool(self.get("allow_short", False))
        use_vol_filter = bool(self.get("volume_filter", False))

        fvg_df = fair_value_gaps(df)
        n = len(df)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        a_bull_top, a_bull_bot, a_bear_top, a_bear_bot = self._active_fvg_levels(
            fvg_df["bull_fvg"].to_numpy(dtype=float),
            fvg_df["bear_fvg"].to_numpy(dtype=float),
            fvg_df["fvg_top"].to_numpy(dtype=float),
            fvg_df["fvg_bottom"].to_numpy(dtype=float),
            high,
            low,
            max_age,
        )

        # Дотик бару до активної зони (bull — своїм low, bear — своїм high).
        touch_bull = np.zeros(n, dtype=bool)
        touch_bear = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.isfinite(a_bull_top[i]) and np.isfinite(a_bull_bot[i]):
                touch_bull[i] = low[i] <= a_bull_top[i] and low[i] >= a_bull_bot[i]
            if np.isfinite(a_bear_top[i]) and np.isfinite(a_bear_bot[i]):
                touch_bear[i] = high[i] >= a_bear_bot[i] and high[i] <= a_bear_top[i]

        # Вхід лише на «свіжому» дотику (перший бар у зоні).
        fresh_bull = touch_bull & ~np.roll(touch_bull, 1)
        fresh_bull[0] = False
        fresh_bear = touch_bear & ~np.roll(touch_bear, 1)
        fresh_bear[0] = False

        if use_vol_filter:
            vol_sma = sma(df["volume"], window=20).to_numpy(dtype=float)
            vol_ok = df["volume"].to_numpy(dtype=float) > vol_sma
            fresh_bull &= vol_ok
            fresh_bear &= vol_ok

        atr14 = atr(df, period=14).to_numpy(dtype=float)
        pos = np.zeros(n, dtype=np.int8)
        sl_price = np.full(n, np.nan)
        tp_price = np.full(n, np.nan)
        cur: int = 0
        cur_sl = cur_tp = float("nan")

        for i in range(n):
            price = float(close[i])
            if cur == 0:
                if fresh_bull[i]:
                    cur = 1
                    cur_sl = price - atr_sl * atr14[i]
                    cur_tp = price + atr_tp * atr14[i]
                elif allow_short and fresh_bear[i]:
                    cur = -1
                    cur_sl = price + atr_sl * atr14[i]
                    cur_tp = price - atr_tp * atr14[i]
            elif cur == 1:
                if fresh_bear[i]:
                    cur = -1 if allow_short else 0
                    if cur == -1:
                        cur_sl = price + atr_sl * atr14[i]
                        cur_tp = price - atr_tp * atr14[i]
                elif low[i] <= cur_sl or high[i] >= cur_tp:
                    cur = 0
            else:  # cur == -1
                if fresh_bull[i]:
                    cur = 1
                    cur_sl = price - atr_sl * atr14[i]
                    cur_tp = price + atr_tp * atr14[i]
                elif high[i] >= cur_sl or low[i] <= cur_tp:
                    cur = 0
            pos[i] = cur
            sl_price[i] = cur_sl if cur != 0 else float("nan")
            tp_price[i] = cur_tp if cur != 0 else float("nan")

        signals = pd.Series(pos, index=df.index, dtype=int)
        levels = pd.DataFrame(index=df.index, dtype=float)
        levels["sl_long"] = np.where(pos == 1, sl_price, np.nan)
        levels["tp_long"] = np.where(pos == 1, tp_price, np.nan)
        levels["sl_short"] = np.where(pos == -1, sl_price, np.nan)
        levels["tp_short"] = np.where(pos == -1, tp_price, np.nan)
        return signals, levels

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        signals, _ = self._run_state_machine(df)
        return signals

    def exit_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рівні SL/TP стан-машини (ATR від ціни входу) — для візуалізації
        та intrabar-симуляції виходів у рушії (intrabar_exits=True)."""
        _, levels = self._run_state_machine(df)
        return levels


__all__ = ["SmcFvgStrategy"]
