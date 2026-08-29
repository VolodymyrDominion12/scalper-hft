"""Децильний lift-аналіз фіч (uplift-концепт).

Джерело: Predictive Marketing (Artun & Levin) — Ch.2, 9.
Аналогія: «surgical discounts» / децильні lift-тести → для кожної фічі рахуємо

    Δ(x) = E[PnL | вхід і X ∈ bin] − E[PnL | вхід]   (на OOS-трейдах)

Фічі з Δ ≈ 0 — кандидати на видалення з ML-фіч-сета; фічі з Δ < 0 у певних
бінах — кандидати на умовні фільтри входу («не торгувати в цьому стані»).

⚠ Confounding: без контролю режиму (волатильність тощо) lift може відбивати
зсув режиму, а не ефект фічі. Для ML-відбору краще порівнювати Δ у межах
одного режиму або додавати режим як контролюючу фічу.

Приклад:
    from scalper_hft.validation.lift import feature_lift_report
    rep = feature_lift_report(trades, features_at_entry)
    print(rep["rsi_14"].to_string())
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decile_lift(
    pnl: pd.Series,
    feature: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Lift за децилями однієї фічі.

    Args:
        pnl: PnL на угоду (Series, індекс спільний з feature).
        feature: значення фічі на вході в угоду.
        n_bins: кількість бінів (децилів).

    Returns:
        DataFrame з колонками:
            bin (нижня межа біна), n, mean_pnl, lift (mean_pnl − overall),
            cum_share (кумулятивна частка PnL).
    """
    df = pd.DataFrame({"pnl": pnl, "feat": feature}).dropna()
    if df.empty or df["feat"].nunique() < 2:
        return pd.DataFrame(columns=["bin", "n", "mean_pnl", "lift", "cum_share"])

    # qcut з fallback на rank (при дублікатах квантилів)
    try:
        df["bin"] = pd.qcut(df["feat"], q=min(n_bins, df["feat"].nunique()), duplicates="drop")
    except ValueError:
        df["bin"] = pd.qcut(df["feat"].rank(method="first"), q=min(n_bins, df["feat"].nunique()),
                            duplicates="drop")

    overall = df["pnl"].mean()
    g = df.groupby("bin", observed=True)["pnl"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_pnl": g.mean(),
            "lift": g.mean() - overall,
        }
    )
    total_pnl = df["pnl"].sum()
    out["cum_share"] = (out["mean_pnl"] * out["n"]).cumsum() / total_pnl if total_pnl != 0 else np.nan
    out = out.reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def feature_lift_report(
    trades: pd.DataFrame,
    features: pd.DataFrame,
    pnl_col: str = "ret",
    feature_cols: list[str] | None = None,
    n_bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Lift-звіт по кількох фічах для набору угод.

    Args:
        trades: DataFrame з колонками ['entry_ts', pnl_col] (формат _extract_trades).
        features: DataFrame значень фіч, індексований за entry_ts (значення на вході).
        pnl_col: колонка PnL на угоду.
        feature_cols: список фіч; None = усі колонки features.
        n_bins: кількість бінів.

    Returns:
        dict: feature → decile_lift DataFrame.
    """
    if trades is None or trades.empty or pnl_col not in trades.columns:
        return {}
    t = trades.copy()
    t["entry_ts"] = pd.to_datetime(t["entry_ts"])
    feats = features.copy()
    feats.index = pd.to_datetime(feats.index)
    merged = t.set_index("entry_ts").join(feats, how="left")

    cols = feature_cols or list(features.columns)
    report: dict[str, pd.DataFrame] = {}
    for col in cols:
        if col not in merged.columns:
            continue
        lift = decile_lift(merged[pnl_col], merged[col], n_bins=n_bins)
        if not lift.empty:
            report[col] = lift
    return report


def lift_summary(report: dict[str, pd.DataFrame], top_k: int = 10) -> pd.DataFrame:
    """Зведення: для кожної фічі — максимальний |lift| і нахил lift по децилях.

    Монотонний lift (нахил далекий від 0) = фіча інформативна;
    |max_lift| ≈ 0 = фічу можна видалити.
    """
    rows: list[dict] = []
    for col, lift in report.items():
        if lift.empty or len(lift) < 2:
            continue
        x = pd.Series(range(len(lift)))
        slope = 0.0
        try:
            from scipy.stats import linregress

            slope = float(linregress(x, lift["mean_pnl"]).slope)
        except Exception:  # noqa: BLE001
            pass
        rows.append(
            {
                "feature": col,
                "max_abs_lift": float(lift["lift"].abs().max()),
                "slope": slope,
                "n_bins": len(lift),
            }
        )
    out = pd.DataFrame(rows).sort_values("max_abs_lift", ascending=False)
    return out.head(top_k)


__all__ = ["decile_lift", "feature_lift_report", "lift_summary"]
