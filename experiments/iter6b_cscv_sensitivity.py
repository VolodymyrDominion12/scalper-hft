"""Iteration 6b — CSCV/PBO + sensitivity для pairs_arb regime_scale.

Підтвердження робастності regime_scale overlay (anti-overfitting skill):
  1. CSCV/PBO на LINK/BTC lb=240 з regime_scale=True: 20 варіантів (entry_z,
     lookback) через run_pairs_backtest → матриця прибутковостей → pbo_cscv.
     PBO < 0.5 = покращення не data-snooping.
  2. Sensitivity по regime_scale_factor (0.0/0.25/0.5/0.75/1.0): плато vs пік.
     smoothness > 0.30 = робастний вибір 0.5.

Запуск: uv run python experiments/iter6b_cscv_sensitivity.py
Вихід: results/iter6b_cscv_sensitivity.csv + звіт у stdout.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scalper_hft.backtest.pairs import run_pairs_backtest
from scalper_hft.config import get_settings
from scalper_hft.data.downloader import download_funding
from scalper_hft.strategies import get_strategy
from scalper_hft.validation.cscv import pbo_cscv

settings = get_settings()

DAYS = 1095
POSITION_PCT = 0.3
N_BLOCKS = 8
N_VARIANTS = 20  # як CSCV_VARIANTS в audit_cell
SEED = 42

# Параметри для CSCV-варіантів (сітка param_space pairs_arb)
ENTRY_Z_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]
LOOKBACK_GRID = [120, 240, 360, 480]
EXIT_Z_GRID = [0.0, 0.3, 0.6]


def load_leg(symbol: str) -> pd.DataFrame:
    from scalper_hft.data.access import ensure_klines

    return ensure_klines(symbol, "1h", DAYS, derive=False)


def pairs_variant_returns(
    d1: pd.DataFrame,
    d2: pd.DataFrame,
    f1: pd.DataFrame,
    f2: pd.DataFrame,
    n_variants: int = N_VARIANTS,
    regime_scale: bool = True,
    factor: float = 0.5,
    seed: int = SEED,
) -> np.ndarray:
    """Матриця прибутковостей (n_variants × n_bars) для випадкових комбінацій
    параметрів pairs_arb через run_pairs_backtest (для CSCV/PBO)."""
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for _ in range(n_variants):
        params = {
            "entry_z": float(rng.choice(ENTRY_Z_GRID)),
            "exit_z": float(rng.choice(EXIT_Z_GRID)),
            "lookback": int(rng.choice(LOOKBACK_GRID)),
            "regime_scale": regime_scale,
            "regime_scale_factor": factor,
        }
        try:
            strat = get_strategy("pairs_arb", **params)
            res = run_pairs_backtest(d1, d2, strat, f1, f2, position_pct=POSITION_PCT, maker_execution=True)
            rows.append(res.equity.pct_change().fillna(0.0).to_numpy(dtype=float))
        except Exception as exc:  # noqa: BLE001
            print(f"  variant {params} ERROR: {exc}", flush=True)
            continue
    if not rows:
        raise RuntimeError("Жоден варіант не відпрацював")
    # вирівняти довжини (можуть відрізнятись якщо dropna)
    min_len = min(len(r) for r in rows)
    return np.vstack([r[:min_len] for r in rows])


def sensitivity_factor(
    d1: pd.DataFrame,
    d2: pd.DataFrame,
    f1: pd.DataFrame,
    f2: pd.DataFrame,
    factors: list[float],
    entry_z: float = 2.0,
    exit_z: float = 0.3,
    lookback: int = 240,
) -> pd.DataFrame:
    """Сітка по regime_scale_factor: метрика = 3y Sharpe (hourly)."""
    rows: list[dict] = []
    for f in factors:
        strat = get_strategy(
            "pairs_arb",
            entry_z=entry_z,
            exit_z=exit_z,
            lookback=lookback,
            regime_scale=True,
            regime_scale_factor=f,
        )
        try:
            res = run_pairs_backtest(d1, d2, strat, f1, f2, position_pct=POSITION_PCT, maker_execution=True)
            rows.append(
                {
                    "factor": f,
                    "ret_3y": round(float(res.metrics.total_return), 4),
                    "maxdd": round(float(res.metrics.max_drawdown), 4),
                    "pf": round(float(res.metrics.profit_factor), 3),
                    "trades": int(res.metrics.n_trades),
                    "sharpe_h": round(float(res.metrics.sharpe_hourly), 5),
                    "calmar": round(float(res.metrics.total_return) / abs(float(res.metrics.max_drawdown) + 1e-9), 3),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"factor": f, "error": str(exc)[:80]})
    return pd.DataFrame(rows)


def smoothness(grid: pd.DataFrame, param: str = "factor") -> float:
    """Гладкість уздовж параметра: 1.0 = плато, ~0 = вузький пік."""
    g = grid.sort_values(param)
    metric = g["sharpe_h"].to_numpy(dtype=float)
    if len(metric) > 2 and metric.std() > 0:
        diffs = np.abs(np.diff(metric))
        return 1.0 - min(1.0, float(diffs.mean() / max(metric.std(), 1e-9)))
    return 1.0


def main() -> None:
    out_csv = Path("results/iter6b_cscv_sensitivity.csv")
    leg1, leg2 = "LINKUSDT", "BTCUSDT"
    print(f"Завантаження даних {leg1}/{leg2} ({DAYS}d)...", flush=True)
    d1, d2 = load_leg(leg1), load_leg(leg2)
    f1, f2 = download_funding(leg1, DAYS), download_funding(leg2, DAYS)

    rows: list[dict] = []

    # ── 1. CSCV/PBO: regime_scale=True vs False ──
    for regime_scale, label in [(True, "regime_scale=True"), (False, "regime_scale=False (baseline)")]:
        print(f"\n[CSCV] {label}: генерація {N_VARIANTS} варіантів...", flush=True)
        t0 = time.time()
        ret = pairs_variant_returns(d1, d2, f1, f2, n_variants=N_VARIANTS, regime_scale=regime_scale, factor=0.5)
        res = pbo_cscv(ret, n_blocks=N_BLOCKS)
        print(res.summary(), flush=True)
        rows.append(
            {
                "test": "CSCV",
                "variant": label,
                "n_variants": res.n_variants,
                "n_combos": res.n_combos,
                "pbo": round(res.pbo, 4),
                "oos_sharpe_median": round(res.oos_sharpe_median, 4),
                "frac_oos_negative": round(float((res.is_best_oos_sharpes < 0).mean()), 3),
                "seconds": round(time.time() - t0, 1),
            }
        )

    # ── 2. Sensitivity по regime_scale_factor ──
    print("\n[Sensitivity] сітка factor ∈ {0.0, 0.25, 0.5, 0.75, 1.0}...", flush=True)
    factors = [0.0, 0.25, 0.5, 0.75, 1.0]
    t0 = time.time()
    grid = sensitivity_factor(d1, d2, f1, f2, factors)
    sm = smoothness(grid)
    print(grid.to_string(index=False), flush=True)
    print(f"smoothness = {sm:.3f}  (>0.30 = плато, робастний)", flush=True)
    for _, r in grid.iterrows():
        row = {"test": "sensitivity", "variant": f"factor={r['factor']}", "smoothness": round(sm, 4)}
        for k in ("ret_3y", "maxdd", "pf", "trades", "sharpe_h", "calmar"):
            if k in r:
                row[k] = r[k]
        rows.append(row)
    rows.append(
        {"test": "sensitivity", "variant": "summary", "smoothness": round(sm, 4), "seconds": round(time.time() - t0, 1)}
    )

    # ── Вердикт ──
    pbo_scaled = next(r for r in rows if r["test"] == "CSCV" and "True" in r["variant"])
    pbo_base = next(r for r in rows if r["test"] == "CSCV" and "baseline" in r["variant"])
    # smoothness на змістовному діапазоні 0.25-1.0 (без degenerate 0.0 = повний блок)
    grid_meaningful = grid[grid["factor"] >= 0.25].reset_index(drop=True)
    sm_meaningful = smoothness(grid_meaningful)
    print("\n=== Вердикт ===", flush=True)
    print(f"PBO regime_scale=True:  {pbo_scaled['pbo']}  ({'PASS' if pbo_scaled['pbo'] < 0.5 else 'FAIL'})")
    print(f"PBO baseline (False):     {pbo_base['pbo']}  ({'PASS' if pbo_base['pbo'] < 0.5 else 'FAIL'})")
    print(f"smoothness factor ∈ [0.0,1.0]:  {sm:.3f}  ({'PASS' if sm > 0.30 else 'FAIL'})")
    print(
        f"smoothness factor ∈ [0.25,1.0]: {sm_meaningful:.3f}  ({'PASS' if sm_meaningful > 0.30 else 'FAIL'}) — без degenerate 0.0"
    )
    best_calmar = grid.loc[grid["calmar"].idxmax()]
    print(
        f"найкращий Calmar при factor={best_calmar['factor']} (Calmar={best_calmar['calmar']}, maxDD={best_calmar['maxdd']})"
    )
    verdict = pbo_scaled["pbo"] < 0.5 and sm_meaningful > 0.30
    print(
        f"\nЗАГАЛЬНИЙ ВЕРДИКТ: {'PASS ✅' if verdict else 'FAIL ⚠'} — "
        f"regime_scale {'робастний' if verdict else 'потребує доопрацювання'}"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
