"""Basis-реверсія: торгівля премією перп-ф'ючерса до споту (delta-neutral).

Логіка:
    - basis_pct = perp_close / spot_close − 1 — премія перпа до споту;
    - basis високий (перп дорожчий за спот) → шорт перпа + лонг споту
      (очікування конвергенції премії);
    - basis низький → дзеркально;
    - вихід: basis повернувся в нейтральну зону.

Доповнює funding_arb: навіть без екстремального фандінгу basis має
середньо-реверсійні властивості (арбітражні потоки). Прибуток = реверсія
basis + зібраний фандінг (якщо carry-напрямок збігається) − 2-leg комісії.

⚠ На Binance basis вузький (1–2.4 bps std) — чесний бектест вирішує,
чи edge покриває комісії.
"""

from __future__ import annotations

import pandas as pd

from scalper_hft.strategies.base import Strategy


class BasisReversion(Strategy):
    name = "basis_reversion"
    family = "relative_value"
    preferred_regimes = frozenset()
    needs_funding = True  # для консистентності рушія; фандінг додається до PnL
    # Потрібні ОБИДВІ ноги (колонки perp+spot). Без цієї декларації стратегія
    # мовчки повертала нулі в будь-якому звичайному klines-прогоні — і 10 клітинок
    # iter7 виглядали як «нульовий edge» замість «немає даних» (аудит 2026-09-11).
    requires = frozenset({"spot_perp"})

    param_space = {
        "exit_bps": (0.0, 10.0, 0.5),
        "lookback": (60.0, 1440.0, 60.0),
    }

    def __init__(
        self,
        exit_bps: float = 2.0,
        lookback: int = 240,
    ) -> None:
        super().__init__(
            exit_bps=exit_bps,
            lookback=int(lookback),
        )

    def generate_signals(
        self, df: pd.DataFrame, trades: pd.DataFrame | None = None, funding: pd.DataFrame | None = None
    ) -> pd.Series:
        """Сигнал пари на основі z-оцінки basis.

        df має містити perp/spot (об'єднаний DataFrame з delta-neutral рушія).
        Вхід: |z| > 2 (basis аномально високий/низький); вихід: |z| < 0.5
        або |basis| < exit_bps. Сигнал на закритті t → виконання t+1.
        """
        if "perp" not in df.columns or "spot" not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)

        basis = df["perp"] / df["spot"] - 1.0
        exit_th = float(self.get("exit_bps", 2.0)) / 10_000.0

        lookback = int(self.get("lookback", 240))
        mean = basis.rolling(lookback, min_periods=lookback // 2).mean()
        std = basis.rolling(lookback, min_periods=lookback // 2).std(ddof=0).replace(0, float("nan"))
        z = ((basis - mean) / std).fillna(0.0)

        sig = pd.Series(float("nan"), index=df.index, dtype=float)
        sig[z > 2.0] = 1.0  # basis високий → шорт перп/лонг спот
        sig[z < -2.0] = -1.0

        prev_pos = sig.ffill().shift(1).fillna(0.0)
        exit_any = (prev_pos != 0.0) & ((basis.abs() < exit_th) | (z.abs() < 0.5))
        sig[exit_any] = 0.0

        return sig.ffill().fillna(0.0).astype(int)
