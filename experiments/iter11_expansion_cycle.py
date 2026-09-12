"""iter11 — незалежне підтвердження ts_momentum long-only.

Pre-registration: docs/reports/hypothesis_iter11_universe.md
НЕ підбирати lookback / smooth / імена після перегляду метрик.

Фази:
    1. variants — CORE / NEW / FULL на 1d + CORE на нативному 4h
    2. leaderboard — docs/reports/LEADERBOARD.md

Запуск:
    uv run python experiments/iter11_expansion_cycle.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter11"
OUT.mkdir(parents=True, exist_ok=True)

CORE_15 = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,"
    "AVAXUSDT,UNIUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,LTCUSDT,AAVEUSDT"
).split(",")

NEW_30 = (
    "BCHUSDT,ETCUSDT,TRXUSDT,XLMUSDT,FILUSDT,VETUSDT,ALGOUSDT,"
    "SNXUSDT,CRVUSDT,COMPUSDT,GRTUSDT,SANDUSDT,MANAUSDT,"
    "AXSUSDT,EGLDUSDT,ICPUSDT,THETAUSDT,APTUSDT,OPUSDT,ARBUSDT,INJUSDT,"
    "SUIUSDT,NEOUSDT,ZECUSDT,DASHUSDT,IOTAUSDT,YFIUSDT"
).split(",")
# EOSUSDT — немає USDT-M лістингу; FTMUSDT/MKRUSDT — валідатор 1d (спайки).
# Пропуск зафіксовано до повторного прогону, не після метрик.

FULL = CORE_15 + NEW_30

# 1d: як iter9/10 (календар ~250/125 днів)
TRAIN_1D, TEST_1D = 250, 125
# 4h: той самий календар (250д × 6 барів, 125д × 6)
TRAIN_4H, TEST_4H = 1500, 750
BARS_PER_YEAR_1D = 365.0
BARS_PER_YEAR_4H = 365.0 * 6.0
NW_LAGS = 20

VARIANTS: list[dict] = [
    {
        "variant": "core15_long_1d",
        "symbols": CORE_15,
        "interval": "1d",
        "allow_short": False,
        "train": TRAIN_1D,
        "test": TEST_1D,
        "bpy": BARS_PER_YEAR_1D,
        "days": 2500,
        "hypothesis": "H0_replication",
    },
    {
        "variant": "new30_long_1d",
        "symbols": NEW_30,
        "interval": "1d",
        "allow_short": False,
        "train": TRAIN_1D,
        "test": TEST_1D,
        "bpy": BARS_PER_YEAR_1D,
        "days": 2500,
        "hypothesis": "H1",
    },
    {
        "variant": "new30_ls_1d",
        "symbols": NEW_30,
        "interval": "1d",
        "allow_short": True,
        "train": TRAIN_1D,
        "test": TEST_1D,
        "bpy": BARS_PER_YEAR_1D,
        "days": 2500,
        "hypothesis": "H1_control",
    },
    {
        "variant": "full45_long_1d",
        "symbols": FULL,
        "interval": "1d",
        "allow_short": False,
        "train": TRAIN_1D,
        "test": TEST_1D,
        "bpy": BARS_PER_YEAR_1D,
        "days": 2500,
        "hypothesis": "H2",
    },
    {
        "variant": "core15_long_4h",
        "symbols": CORE_15,
        "interval": "4h",
        "allow_short": False,
        "train": TRAIN_4H,
        "test": TEST_4H,
        "bpy": BARS_PER_YEAR_4H,
        "days": 2500,
        "hypothesis": "H3",
    },
    {
        "variant": "core15_ls_4h",
        "symbols": CORE_15,
        "interval": "4h",
        "allow_short": True,
        "train": TRAIN_4H,
        "test": TEST_4H,
        "bpy": BARS_PER_YEAR_4H,
        "days": 2500,
        "hypothesis": "H3_control",
    },
]


def newey_west_t(returns: np.ndarray, lags: int = NW_LAGS) -> float:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3:
        return 0.0
    mu = float(r.mean())
    dev = r - mu
    gamma0 = float((dev @ dev) / n)
    var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        cov = float((dev[lag:] @ dev[:-lag]) / n)
        var += 2.0 * w * cov
    if var <= 0:
        return 0.0
    return float(mu / np.sqrt(var / n))


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    bpy: float,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    p = 1.0 / 10.0
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = int(rng.integers(n))
        for k in range(n):
            idx[k] = i
            if rng.random() < p:
                i = int(rng.integers(n))
            else:
                i = (i + 1) % n
        s = r[idx]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd * np.sqrt(bpy) if sd > 0 else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _period_sharpe(port: pd.Series, bpy: float, start: str, end: str) -> float:
    sl = port.loc[start:end].dropna()
    if len(sl) < 30:
        return float("nan")
    sd = sl.std(ddof=1)
    return float(sl.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0


def _portfolio_test(variant: dict, *, vol_window: int = 20) -> dict:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.walk_forward import run_walk_forward

    interval = str(variant["interval"])
    train = int(variant["train"])
    test = int(variant["test"])
    bpy = float(variant["bpy"])
    days = int(variant["days"])
    strategy = get_strategy(
        "ts_momentum",
        lookback=20,
        signal_smooth=1,
        allow_short=bool(variant["allow_short"]),
    )
    cost = CostModel.from_settings(get_settings())
    per_symbol: dict[str, pd.Series] = {}
    wf_sharpes: list[float] = []
    skipped: list[str] = []
    n_trades = 0

    for symbol in list(variant["symbols"]):
        try:
            df = ensure_klines(symbol, interval, days, derive=False)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{symbol}: {type(exc).__name__}")
            continue
        if df is None or len(df) < train + test + 10:
            skipped.append(f"{symbol}: bars={0 if df is None else len(df)}")
            continue
        wf = run_walk_forward(
            df,
            strategy,
            train_bars=train,
            test_bars=test,
            cost=cost,
            is_maker=True,
            interval=interval,
            collect_oos_returns=True,
            purge_bars=2,
            embargo_bars=2,
            strict_data=False,
        )
        oos = wf.oos_returns.dropna()
        if oos.empty:
            skipped.append(f"{symbol}: empty OOS")
            continue
        per_symbol[symbol] = oos
        wf_sharpes.append(float(wf.avg_oos_sharpe))
        n_trades += int(sum(w.n_trades for w in wf.windows))

    if not per_symbol:
        return {
            "variant": variant["variant"],
            "hypothesis": variant["hypothesis"],
            "strategy": "ts_momentum",
            "interval": interval,
            "error": "no data",
            "skipped": "; ".join(skipped[:8]),
        }

    mat = pd.DataFrame(per_symbol)
    port = mat.mean(axis=1).dropna()
    sd = port.std(ddof=1)
    port_sharpe = float(port.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0
    t_nw = newey_west_t(port.to_numpy())
    ci_lo, ci_hi = bootstrap_sharpe_ci(port.to_numpy(), bpy)

    realized = port.rolling(vol_window, min_periods=20).std(ddof=0).shift(1)
    med_vol = float(realized.median())
    w = (med_vol / realized).clip(0.0, 2.0).fillna(0.0) if med_vol > 0 else pd.Series(1.0, index=port.index)
    port_vt = (port * w).dropna()
    sd_vt = port_vt.std(ddof=1)
    vt_sharpe = float(port_vt.mean() / sd_vt * np.sqrt(bpy)) if sd_vt > 0 else 0.0
    t_vt = newey_west_t(port_vt.to_numpy())

    corr = mat.corr()
    n_sym = corr.shape[0]
    mean_pair_corr = float("nan")
    if n_sym >= 2:
        mask = np.triu(np.ones((n_sym, n_sym), dtype=bool), k=1)
        mean_pair_corr = float(corr.where(mask).stack().mean())

    pos_frac = float(np.mean([s > 0 for s in wf_sharpes])) if wf_sharpes else 0.0
    h1_pass = bool(port_sharpe >= 0.5 and t_nw >= 2.0)
    return {
        "variant": variant["variant"],
        "hypothesis": variant["hypothesis"],
        "strategy": "ts_momentum",
        "interval": interval,
        "allow_short": bool(variant["allow_short"]),
        "n_symbols": len(per_symbol),
        "n_requested": len(variant["symbols"]),
        "mean_oos_wf": float(np.mean(wf_sharpes)) if wf_sharpes else 0.0,
        "symbols_pos_frac": pos_frac,
        "n_trades_oos": n_trades,
        "port_sharpe": port_sharpe,
        "t_newey_west": t_nw,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "port_sharpe_vt": vt_sharpe,
        "t_newey_west_vt": t_vt,
        "mean_pair_corr": mean_pair_corr,
        "sharpe_2019_2022": _period_sharpe(port, bpy, "2019-01-01", "2022-12-31"),
        "sharpe_2023_2026": _period_sharpe(port, bpy, "2023-01-01", "2026-12-31"),
        "gate_pass": h1_pass,
        "skipped": "; ".join(skipped[:12]),
        "notes": (
            f"{variant['hypothesis']}; short={variant['allow_short']}; "
            f"n={len(per_symbol)}/{len(variant['symbols'])} {interval}"
        ),
    }


def phase_variants() -> pd.DataFrame:
    print("=== iter11: pre-registered variants ===", flush=True)
    rows: list[dict] = []
    for i, var in enumerate(VARIANTS, 1):
        t0 = time.time()
        row = _portfolio_test(var)
        rows.append(row)
        print(
            f"[{i}/{len(VARIANTS)}] {var['variant']:<20s} "
            f"n={row.get('n_symbols', 0):>2} "
            f"port_SR={row.get('port_sharpe', 0):+.3f} "
            f"t_NW={row.get('t_newey_west', 0):+.2f} "
            f"CI=[{row.get('ci_lo', float('nan')):+.2f},{row.get('ci_hi', float('nan')):+.2f}] "
            f"gate={'PASS' if row.get('gate_pass') else 'FAIL'} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if row.get("skipped"):
            print(f"    skip: {row['skipped'][:200]}", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "variants.csv"
    df.to_csv(out_csv, index=False)
    print(f"Збережено: {out_csv}", flush=True)
    return df


def phase_leaderboard() -> str:
    from scalper_hft.research.leaderboard import leaderboard_dataframe, render_leaderboard_markdown

    md = render_leaderboard_markdown(ROOT / "results")
    out_md = ROOT / "docs" / "reports" / "LEADERBOARD.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    leaderboard_dataframe(ROOT / "results").to_csv(OUT / "leaderboard.csv", index=False)
    print(f"Збережено: {out_md}", flush=True)
    return str(out_md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "variants", "leaderboard"], default="all")
    args = ap.parse_args()
    if args.phase in ("all", "variants"):
        phase_variants()
    if args.phase in ("all", "leaderboard"):
        phase_leaderboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
