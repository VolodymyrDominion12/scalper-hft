"""Iteration 8 — валідація regime-карти v2.0 на СВІЖИХ символах (поза спаленим OOS).

Питання: чи переноситься емпірична карта «режим → стратегія» (iter7) на
інструменти й період, які не використовувались при її побудові?

Дизайн (без lookahead):
    - символи: ATOMUSDT, DOTUSDT, LTCUSDT, NEARUSDT (НЕ входили в iter7);
    - fit: 2023-09-11 → 2025-09-11 (2 роки, IS) на цих же 4 символах → карта;
    - apply: 2025-09-11 → 2026-09-11 (1 рік) walk-forward train=2500/test=500,
      maker — жоден бар apply-періоду карта не бачила.

Порівнюємо (усі на одній сітці символів×вікон):
    map.v2         — RegimeSupervisor + regime_map_path (слеви iter7)
    map.v2:bp      — те саме, blend_mode=best_prior
    map.v2:dwell3  — те саме, min_dwell_bars=3 (кандидат у дефолт)
    taxonomy.v1.4  — ті самі діти БЕЗ карти (пріори preferred_regimes)
    single:*       — три слеви окремо
    equal_weight   — ensemble mode=mean тих самих трьох

Запуск: .venv/bin/python experiments/iter8_regime_map_validation.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["ATOMUSDT", "DOTUSDT", "LTCUSDT", "NEARUSDT"]
INTERVAL = "1h"
BASE = "1m"
FIT_END = "2025-09-11"
FIT_DAYS = 730
APPLY_DAYS = 365
CHILDREN = "supertrend,stoch_rsi,funding_carry"
OUT_CSV = Path("results/iter8_validation.csv")
OUT_MAP = Path("results/iter8_regime_map.json")
PPY = 8760.0


def build_map() -> Path:
    """Fit карти на IS-періоді свіжих символів → JSON."""
    from scalper_hft.validation.regime_fit import fit_regime_map, format_fit_summary
    from scalper_hft.validation.regime_map import save_regime_map

    res = fit_regime_map(
        CHILDREN.split(","),
        SYMBOLS,
        interval=INTERVAL,
        fit_days=FIT_DAYS,
        fit_end=FIT_END,
        base_interval=BASE,
        is_maker=True,
    )
    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    save_regime_map(res.map, OUT_MAP, meta=res.meta)
    print(format_fit_summary(res), flush=True)
    return OUT_MAP


def variants(map_path: Path) -> list[tuple[str, dict]]:
    v: list[tuple[str, dict]] = [
        (
            "map.v2",
            {
                "name": "regime_supervisor",
                "children": CHILDREN,
                "kwargs": {"blend_mode": "regime_soft", "regime_map_path": str(map_path)},
            },
        ),
        (
            "map.v2:bp",
            {
                "name": "regime_supervisor",
                "children": CHILDREN,
                "kwargs": {"blend_mode": "best_prior", "regime_map_path": str(map_path)},
            },
        ),
        (
            "map.v2:dwell3",
            {
                "name": "regime_supervisor",
                "children": CHILDREN,
                "kwargs": {"blend_mode": "regime_soft", "regime_map_path": str(map_path), "min_dwell_bars": 3},
            },
        ),
        (
            "taxonomy.v1.4",
            {"name": "regime_supervisor", "children": CHILDREN, "kwargs": {"blend_mode": "regime_soft"}},
        ),
        ("single:supertrend", {"name": "supertrend", "kwargs": {}}),
        ("single:stoch_rsi", {"name": "stoch_rsi", "kwargs": {}}),
        ("single:funding_carry", {"name": "funding_carry", "kwargs": {}}),
        ("equal_weight", {"name": "ensemble", "children": CHILDREN, "mode": "mean"}),
    ]
    return v


def _build(spec: dict):
    from scalper_hft.strategies import get_strategy

    if spec["name"] == "regime_supervisor":
        return get_strategy("regime_supervisor", strategies=spec["children"], n_hmm_states=3, hmm_fit_bars=2000, **spec["kwargs"])
    if spec["name"] == "ensemble":
        return get_strategy("ensemble", strategies=spec["children"], mode=spec["mode"])
    return get_strategy(spec["name"], **spec.get("kwargs", {}))


def _sharpe(r: pd.Series, min_bars: int = 100) -> float:
    r = pd.Series(r).dropna()
    if len(r) < min_bars:
        return float("nan")
    sd = float(r.std(ddof=0))
    return float(r.mean() / sd * np.sqrt(PPY)) if sd > 1e-12 else 0.0


def run_cell(symbol: str, label: str, spec: dict, train: int, test: int, maker: bool) -> dict:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.store import get_store
    from scalper_hft.validation.walk_forward import run_walk_forward

    row: dict = {"symbol": symbol, "variant": label, "status": "ok", "error": ""}
    t0 = time.time()
    try:
        settings = get_settings()
        strat = _build(spec)
        df = ensure_klines(symbol, INTERVAL, APPLY_DAYS, base_interval=BASE, derive=True, readonly=True)
        store = get_store()
        trades = store.load_trades(symbol) if strat.needs_trades else None
        funding = store.load_funding(symbol) if strat.needs_funding else None
        cost = CostModel(
            maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac
        )
        res = run_walk_forward(
            df,
            strat,
            train,
            test,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
            interval=INTERVAL,
            is_maker=maker,
            collect_oos_returns=True,
        )
    except Exception as exc:  # noqa: BLE001
        row.update(status="error", error=str(exc)[:250], seconds=round(time.time() - t0, 1))
        return row

    oos = res.oos_returns
    row.update(
        {
            "n_bars": int(len(df)),
            "n_windows": len(res.windows),
            "n_trades_oos": int(sum(w.n_trades for w in res.windows)),
            "avg_oos_sharpe": round(float(res.avg_oos_sharpe), 4),
            "oos_positive_frac": round(float(res.positive_windows_frac), 4),
            "seconds": round(time.time() - t0, 1),
        }
    )
    if oos is not None and not oos.empty:
        r = pd.Series(oos).astype(float)
        eq = (1 + r).cumprod()
        row["pooled_oos_sharpe"] = round(_sharpe(r), 4)
        row["oos_total_return"] = round(float(eq.iloc[-1] - 1), 4)
        row["oos_max_dd"] = round(float((eq / eq.cummax() - 1).min()), 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--reuse-map", action="store_true", help="Не перебудовувати карту, взяти results/iter8_regime_map.json")
    args = ap.parse_args()

    map_path = OUT_MAP if (args.reuse_map and OUT_MAP.exists()) else build_map()
    vs = variants(map_path)

    rows: list[dict] = []
    t0 = time.time()
    for symbol in SYMBOLS:
        for label, spec in vs:
            rows.append(run_cell(symbol, label, spec, args.train, args.test, True))
            print(f"  {symbol}/{label} готово ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    ok = df[df["status"] == "ok"].copy()
    ok["is_pos"] = (ok["avg_oos_sharpe"] > 0).astype(float)
    ok["pass_gate"] = ((ok["avg_oos_sharpe"] > 0.3) & (ok["oos_positive_frac"] >= 0.5)).astype(float)
    agg = (
        ok.groupby("variant")
        .agg(
            n_symbols=("symbol", "count"),
            mean_oos_sharpe=("avg_oos_sharpe", "mean"),
            median_oos_sharpe=("avg_oos_sharpe", "median"),
            mean_pooled_sharpe=("pooled_oos_sharpe", "mean"),
            mean_pos_frac=("oos_positive_frac", "mean"),
            mean_ret=("oos_total_return", "mean"),
            worst_dd=("oos_max_dd", "min"),
            n_trades=("n_trades_oos", "sum"),
            frac_symbols_positive=("is_pos", "mean"),
            frac_pass_gate=("pass_gate", "mean"),
        )
        .reset_index()
        .sort_values("mean_pooled_sharpe", ascending=False)
    )
    pd.set_option("display.width", 220)
    print("\n=== iter8: apply-період 2025-09-11 → 2026-09-11 (свіжі символи) ===")
    print(agg.round(3).to_string(index=False))
    print("\nper-symbol:")
    piv = ok.pivot_table(index="variant", columns="symbol", values="avg_oos_sharpe", aggfunc="mean")
    print(piv.round(3).to_string())
    agg.to_csv(OUT_CSV.with_name("iter8_validation_summary.csv"), index=False)
    print(f"\nCSV: {OUT_CSV}\nMap: {map_path}")


if __name__ == "__main__":
    main()
