"""SMC/ICT хелпери (перенесено з trade-bot-main/smc).

Поки що тут лише те, що потрібно портованим стратегіям:
    - `fair_value_gaps()` — Fair Value Gaps (зони дисбалансу).

Усі функції — без lookahead: використовують лише дані до закриття поточного
бару (формування FVG на барі t бачить high/low бару t та high/low бару t-2;
вхід у зону можливий лише на пізніших барах).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Fair Value Gaps — зони дисбалансу ціни.

    - bull_fvg: мінімум поточної свічки вищий за максимум свічки [i-2]
      (бичачий розрив — «магніт» для повернення ціни зверху);
    - bear_fvg: максимум поточної свічки нижчий за мінімум свічки [i-2].

    Returns:
        DataFrame з колонками:
            bull_fvg (int)   — 1 якщо на барі сформувався бичачий FVG;
            bear_fvg (int)   — 1 якщо на барі сформувався ведмежий FVG;
            fvg_top (float)  — верхня межа зони;
            fvg_bottom (float) — нижня межа зони.
    """
    high_shift2 = df["high"].shift(2)
    low_shift2 = df["low"].shift(2)

    bull_fvg = df["low"] > high_shift2
    bear_fvg = df["high"] < low_shift2

    fvg_top = pd.Series(np.nan, index=df.index, dtype=float)
    fvg_bottom = pd.Series(np.nan, index=df.index, dtype=float)

    fvg_top.loc[bull_fvg] = df["low"].loc[bull_fvg]
    fvg_bottom.loc[bull_fvg] = high_shift2.loc[bull_fvg]

    fvg_top.loc[bear_fvg] = low_shift2.loc[bear_fvg]
    fvg_bottom.loc[bear_fvg] = df["high"].loc[bear_fvg]

    return pd.DataFrame(
        {
            "bull_fvg": bull_fvg.astype(int),
            "bear_fvg": bear_fvg.astype(int),
            "fvg_top": fvg_top,
            "fvg_bottom": fvg_bottom,
        }
    )


__all__ = ["fair_value_gaps"]
