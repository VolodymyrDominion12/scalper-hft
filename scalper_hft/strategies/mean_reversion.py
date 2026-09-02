"""Mean-reversion скальпер: RSI + Bollinger на свічках.

Логіка (книга, гл. 3 — mean reversion; гл. 5 — такі стратегії найменш
чутливі до slippage):
    - вхід у лонг, коли RSI < oversold_threshold І ціна нижче нижньої смуги BB
      (перепродаж), за умови достатньої волатильності (volatility floor);
    - вхід у шорт, коли RSI > overbought_threshold І ціна вище верхньої смуги;
    - вихід: повернення до середини (RSI ~ 50) або жорсткий стоп-лос.
"""

from __future__ import annotations

import pandas as pd

from scalper_hft.features.indicators import add_standard_features
from scalper_hft.strategies.base import Strategy


class MeanReversionScalper(Strategy):
    name = "mean_reversion"

    param_space = {
        "rsi_period": (5.0, 30.0, 1.0),
        "oversold": (10.0, 40.0, 2.0),
        "overbought": (60.0, 90.0, 2.0),
        "bb_period": (10.0, 60.0, 5.0),
        "min_atr_pct": (0.0005, 0.005, 0.0005),
        "stop_atr_mult": (1.0, 4.0, 0.5),
        "max_trend": (0.2, 1.0, 0.1),
    }

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        bb_period: int = 20,
        min_atr_pct: float = 0.001,
        stop_atr_mult: float = 2.0,
        max_trend: float = 0.6,
        skip_high_vol: bool = True,
    ) -> None:
        super().__init__(
            rsi_period=int(rsi_period),
            oversold=oversold,
            overbought=overbought,
            bb_period=int(bb_period),
            min_atr_pct=min_atr_pct,
            stop_atr_mult=stop_atr_mult,
            max_trend=max_trend,
            skip_high_vol=skip_high_vol,
        )

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        from scalper_hft.features.indicators import rsi
        from scalper_hft.features.regimes import trend_strength, volatility_regime

        f = add_standard_features(df)
        rsi_val = rsi(f["close"], int(self.get("rsi_period", 14)))
        close = f["close"]
        mid = f["bb_mid"]
        atr_pct = f["atr_14"] / close.replace(0, float("nan"))

        long_entry = (rsi_val < self.get("oversold", 30.0)) & (close < f["bb_low"])
        short_entry = (rsi_val > self.get("overbought", 70.0)) & (close > f["bb_up"])
        vol_ok = atr_pct >= self.get("min_atr_pct", 0.001)

        # ── regime-фільтри (гл. 4/10 книги; features/regimes.py) ────────────
        # mean reversion працює у флеті: входимо лише коли тренд слабкий
        trend = trend_strength(close, 9, 50)
        trend_ok = trend < float(self.get("max_trend", 0.6))
        # і волатильність не в режимі 'high' (вибухові рухи ≠ реверсія)
        if self.get("skip_high_vol", True):
            regime = volatility_regime(close, lookback=60, percentile_window=500)
            vol_ok = vol_ok & (regime != "high")

        # Патерн "вхід утримується до виходу": у sig лише входи, виходи
        # прив'язані до ПОПЕРЕДНЬОГО стану позиції (без lookahead).
        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[long_entry & vol_ok & trend_ok] = 1.0
        sig[short_entry & vol_ok & trend_ok] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        stop = f["atr_14"] * self.get("stop_atr_mult", 2.0)
        # вихід лонга: ціна повернулась до середини або пробила стоп
        exit_long = (prev_pos == 1.0) & ((close >= mid) | (close < mid - stop))
        # вихід шорта: дзеркально
        exit_short = (prev_pos == -1.0) & ((close <= mid) | (close > mid + stop))
        sig[exit_long | exit_short] = 0.0

        return sig.ffill().fillna(0.0).astype(int)

    def generate_signals_traced(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ):
        """Детальний трейсинг: записує кожен потенційний сигнал і причину блокування.

        Для кожного бару, де raw-логіка генерує long_entry або short_entry,
        перевіряємо всі фільтри окремо і записуємо які саме заблокували угоду.
        """
        from scalper_hft.features.indicators import rsi
        from scalper_hft.features.regimes import trend_strength, volatility_regime
        from scalper_hft.research.filter_trace import FilterTrace, SignalEvent

        f = add_standard_features(df)
        rsi_val = rsi(f["close"], int(self.get("rsi_period", 14)))
        close = f["close"]
        mid = f["bb_mid"]
        atr_pct = f["atr_14"] / close.replace(0, float("nan"))

        long_entry = (rsi_val < self.get("oversold", 30.0)) & (close < f["bb_low"])
        short_entry = (rsi_val > self.get("overbought", 70.0)) & (close > f["bb_up"])

        # Обчислюємо кожен фільтр окремо для трейсингу
        vol_filter = atr_pct >= self.get("min_atr_pct", 0.001)
        trend = trend_strength(close, 9, 50)
        trend_filter = trend < float(self.get("max_trend", 0.6))

        if self.get("skip_high_vol", True):
            regime = volatility_regime(close, lookback=60, percentile_window=500)
            vol_regime_filter = regime != "high"
        else:
            vol_regime_filter = pd.Series(True, index=df.index)

        vol_ok = vol_filter & vol_regime_filter

        # Генеруємо фінальні сигнали (стандартний шлях)
        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[long_entry & vol_ok & trend_filter] = 1.0
        sig[short_entry & vol_ok & trend_filter] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        stop = f["atr_14"] * self.get("stop_atr_mult", 2.0)
        exit_long = (prev_pos == 1.0) & ((close >= mid) | (close < mid - stop))
        exit_short = (prev_pos == -1.0) & ((close <= mid) | (close > mid + stop))
        sig[exit_long | exit_short] = 0.0
        final_signals = sig.ffill().fillna(0.0).astype(int)

        # ── Збираємо FilterTrace ─────────────────────────────────────────────
        trace = FilterTrace()
        raw_signal_mask = long_entry | short_entry
        for ts in df.index[raw_signal_mask]:
            raw = 1 if long_entry.get(ts, False) else -1
            blocked: list[str] = []
            if not vol_filter.get(ts, True):
                blocked.append("vol_ok (atr_pct)")
            if not vol_regime_filter.get(ts, True):
                blocked.append("vol_ok (high_regime)")
            if not trend_filter.get(ts, True):
                blocked.append("trend_ok")

            final = final_signals.get(ts, 0)
            rsi_v = rsi_val.get(ts, float("nan"))
            atr_v = atr_pct.get(ts, float("nan"))
            trend_v = trend.get(ts, float("nan"))
            trace.add(
                SignalEvent(
                    ts=ts,
                    raw_signal=raw,
                    final_signal=int(final) if not blocked else 0,
                    blocked_by=blocked,
                    context={
                        "rsi": round(float(rsi_v), 2) if rsi_v == rsi_v else None,
                        "atr_pct": round(float(atr_v), 6) if atr_v == atr_v else None,
                        "trend": round(float(trend_v), 4) if trend_v == trend_v else None,
                    },
                )
            )

        return final_signals, trace

    def exit_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рівні виходу для візуалізації: ціль = середина BB, стоп = середина ∓ ATR×mult.

        Лонг: tp = bb_mid (повернення до середини), sl = bb_mid − atr×mult;
        шорт — дзеркально. Збігається з логікою виходів у generate_signals.
        Увага: рівні відносні СЕРЕДИНИ смуг, а не ціни входу — стратегія
        входить глибоко за bb_low/bb_up, тому sl може опинитись вище входу
        (лонг) — це чесна картина фактичної логіки виходу, не баг.
        """
        from scalper_hft.features.indicators import add_standard_features

        f = add_standard_features(df)
        stop = f["atr_14"] * self.get("stop_atr_mult", 2.0)
        out = pd.DataFrame(index=df.index, dtype=float)
        out["sl_long"] = f["bb_mid"] - stop
        out["tp_long"] = f["bb_mid"]
        out["sl_short"] = f["bb_mid"] + stop
        out["tp_short"] = f["bb_mid"]
        return out
