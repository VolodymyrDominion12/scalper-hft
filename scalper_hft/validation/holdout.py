"""«Замкований» holdout (Narang гл. 9; overfitting-audit SKILL): останні `holdout_pct`%
даних НЕ використовуються для підбору параметрів — лише для фінального сліпого тесту.

Проблема, яку вирішує модуль: без зарезервованого holdout дослідник
повторно торкається OOS при кожній ітерації і потроху «спалює» його підбором,
самі того не помічаючи. Holdout залишає шматок історії недоторканим, поки не
прийде час фінального рішення «пускати в live чи ні».

Поведінка:
    split_research_holdout(df, holdout_pct) -> (research_df, holdout_df)
        - research_df: перші (100 - holdout_pct)% барів (для WF/Optuna/sensitivity/DSR);
        - holdout_df:   останні holdout_pct% барів (сліпий фінальний тест).
    holdout_pct <= 0 → (df, empty) (вимкнено, поточна поведінка).
"""

from __future__ import annotations

import pandas as pd


def split_research_holdout(df: pd.DataFrame, holdout_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Поділити ряд на research (перші 100-holdout_pct %) та holdout (останні holdout_pct %).

    holdout_pct у частках (0.0–1.0) або відсотках (0–100): значення >1 трактуємо як
    відсотки. holdout_pct <= 0 → holdout вимкнено (повертаємо весь df).
    Ділиться за кількістю барів (не за календарем), щоб бути детермінованим.
    """
    if df is None or df.empty or holdout_pct <= 0:
        return df, df.iloc[0:0]  # порожній holdout тієї ж форми
    pct = float(holdout_pct)
    if pct > 1.0:  # відсотки → частка
        pct = pct / 100.0
    pct = min(max(pct, 0.0), 0.95)  # не більше 95% holdout
    n_hold = int(round(len(df) * pct))
    if n_hold <= 0:
        return df, df.iloc[0:0]
    if n_hold >= len(df):
        # весь ряд — holdout: research порожній (некоректний конфіг, але без raise)
        return df.iloc[0:0], df
    research = df.iloc[:-n_hold]
    holdout = df.iloc[-n_hold:]
    return research, holdout


__all__ = ["split_research_holdout"]
