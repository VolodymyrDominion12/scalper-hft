"""Time-series momentum (TSMOM) — власний momentum активу за lookback барів.

Навіщо окрема стратегія. У `cross_momentum` є гілка «один символ → rolling
percentile», але capability contract навмисно її блокує: у матричному прогоні
клітинка з назвою `cross_momentum` рахувала б ІНШИЙ механізм (часовий, не
крос-секційний) — це тиха підміна альфи. Тут той самий механізм винесено під
власним ім'ям, щоб дослідження momentum на денних барах було чесним.

Механізм (Moskowitz–Ooi–Pedersen, time-series momentum; Narang гл. 3):
    - ret_t = close_t / close_{t−lookback} − 1 — накопичений імпульс;
    - вхід у лонг, якщо ret_t перевищує верхній квантиль власного rolling
      розподілу ret (вікно lookback×quantile_window), у шорт — якщо нижчий;
    - `signal_smooth` — EMA-згладжування позиції (менше переворотів);
    - поза смугами — 0 (поза ринком).

Без lookahead: усі вікна закриті праворуч (квантиль і повернення — на барі t),
лаг виконання робить рушій бектесту (shift(1)).

Економіка: сигнал розраховано на довгі горизонти (1d/4h), де round-trip
вартість 10 bps (taker) або 4 bps + slippage (maker) мала відносно типового
хвильового руху — на 1m/5m/15m такий momentum повністю з'їдається комісіями
(див. docs/reports/iter9_*.md).
"""

from __future__ import annotations

import pandas as pd

from scalper_hft.strategies.base import Strategy


class TimeSeriesMomentum(Strategy):
    name = "ts_momentum"
    family = "momentum"
    # momentum-механізм дає edge в обидві структурні фази (long у trend_up,
    # short у trend_down); тег — гіпотеза для research, не live-гейт.
    preferred_regimes = frozenset({"trend_up", "trend_down"})

    param_space = {
        "lookback": (5.0, 60.0, 5.0),
        "top_pct": (0.1, 0.4, 0.1),
        "quantile_window": (2.0, 6.0, 1.0),
        "signal_smooth": (1.0, 5.0, 1.0),
    }

    def __init__(
        self,
        lookback: int = 20,
        top_pct: float = 0.2,
        quantile_window: int = 3,
        signal_smooth: int = 1,
        allow_short: bool = True,
    ) -> None:
        super().__init__(
            lookback=int(lookback),
            top_pct=float(top_pct),
            quantile_window=int(quantile_window),
            signal_smooth=int(signal_smooth),
            allow_short=bool(allow_short),
        )

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        lookback = max(int(self.get("lookback", 20)), 1)
        top_pct = min(max(float(self.get("top_pct", 0.2)), 0.01), 0.5)
        qwin = max(int(self.get("quantile_window", 3)), 1)
        smooth = max(int(self.get("signal_smooth", 1)), 1)
        allow_short = bool(self.get("allow_short", True))

        close = df["close"].astype(float)
        ret = close.pct_change(lookback)
        window = max(lookback * qwin, lookback + 2)
        high = ret.rolling(window, min_periods=max(lookback, 5)).quantile(1.0 - top_pct)
        low = ret.rolling(window, min_periods=max(lookback, 5)).quantile(top_pct)

        sig = pd.Series(0.0, index=df.index, dtype=float)
        sig[ret > high] = 1.0
        if allow_short:
            sig[ret < low] = -1.0
        if smooth > 1:
            sig = sig.ewm(span=smooth, adjust=False).mean()
            sig = sig.apply(lambda v: 1.0 if v > 0.3 else (-1.0 if v < -0.3 else 0.0))
        return sig.fillna(0.0).clip(-1, 1).astype(int)


__all__ = ["TimeSeriesMomentum"]
