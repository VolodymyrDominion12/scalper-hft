"""iter10 — цикл дослідження: покращення ts_momentum → аудит → лідерборд.

Фази:
    1. variants — тест варіантів параметрів (менше turnover, long-only, vol-target)
    2. audit — вибірковий аудит топ-комірок ts_momentum 1d
    3. leaderboard — генерація docs/reports/LEADERBOARD.md

Запуск:
    uv run python experiments/iter10_research_cycle.py --phase all
    uv run python experiments/iter10_research_cycle.py --phase variants
    uv run python experiments/iter10_research_cycle.py --phase audit --no-cscv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter10"
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,"
    "AVAXUSDT,UNIUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,LTCUSDT,AAVEUSDT"
).split(",")

# Символи з позитивним WF OOS на 6-річній історії (iter9 daily_long_history)
TOP10_SYMBOLS = [
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "ETHUSDT",
    "BTCUSDT",
]

TRAIN, TEST = 250, 125
BARS_PER_YEAR = 365.0
NW_LAGS = 20

# Варіанти для тесту гіпотез iter9 (fee-drag, turnover, bull-bias)
VARIANTS: list[dict] = [
    {"variant": "baseline_lb20", "lookback": 20, "signal_smooth": 1, "allow_short": True, "symbols": SYMBOLS},
    {"variant": "smooth3_lb20", "lookback": 20, "signal_smooth": 3, "allow_short": True, "symbols": SYMBOLS},
    {"variant": "long_only_lb20", "lookback": 20, "signal_smooth": 1, "allow_short": False, "symbols": SYMBOLS},
    {"variant": "smooth3_long_lb20", "lookback": 20, "signal_smooth": 3, "allow_short": False, "symbols": SYMBOLS},
    {"variant": "top10_lb20", "lookback": 20, "signal_smooth": 1, "allow_short": True, "symbols": TOP10_SYMBOLS},
    {"variant": "smooth3_top10", "lookback": 20, "signal_smooth": 3, "allow_short": True, "symbols": TOP10_SYMBOLS},
]

AUDIT_CELLS: list[tuple[str, str, str]] = [
    ("ts_momentum", "DOGEUSDT", "1d"),
    ("ts_momentum", "ADAUSDT", "1d"),
    ("ts_momentum", "AVAXUSDT", "1d"),
    ("ts_momentum", "NEARUSDT", "1d"),
    ("ts_momentum", "DOTUSDT", "1d"),
    ("ts_momentum", "ATOMUSDT", "1d"),
    ("ts_momentum", "BTCUSDT", "1d"),
    ("ts_momentum", "BNBUSDT", "1d"),
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


def _portfolio_test(
    variant: dict,
    *,
    days: int = 2500,
    vol_window: int = 20,
) -> dict:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.walk_forward import run_walk_forward

    strategy = get_strategy(
        "ts_momentum",
        lookback=int(variant["lookback"]),
        signal_smooth=int(variant["signal_smooth"]),
        allow_short=bool(variant["allow_short"]),
    )
    cost = CostModel.from_settings(get_settings())
    symbols = list(variant["symbols"])
    per_symbol: dict[str, pd.Series] = {}
    wf_sharpes: list[float] = []
    n_trades = 0

    for symbol in symbols:
        try:
            df = ensure_klines(symbol, "1d", days, derive=False)
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < TRAIN + TEST + 10:
            continue
        wf = run_walk_forward(
            df,
            strategy,
            train_bars=TRAIN,
            test_bars=TEST,
            cost=cost,
            is_maker=True,
            interval="1d",
            collect_oos_returns=True,
            purge_bars=2,
            embargo_bars=2,
            strict_data=False,
        )
        oos = wf.oos_returns.dropna()
        if oos.empty:
            continue
        per_symbol[symbol] = oos
        wf_sharpes.append(float(wf.avg_oos_sharpe))
        n_trades += int(sum(w.n_trades for w in wf.windows))

    if not per_symbol:
        return {"variant": variant["variant"], "error": "no data"}

    mat = pd.DataFrame(per_symbol)
    port = mat.mean(axis=1).dropna()
    sd = port.std(ddof=1)
    port_sharpe = float(port.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    t_nw = newey_west_t(port.to_numpy())

    # vol-target overlay (iter9 §6.1)
    realized = port.rolling(vol_window, min_periods=20).std(ddof=0).shift(1)
    med_vol = float(realized.median())
    w = (med_vol / realized).clip(0.0, 2.0).fillna(0.0) if med_vol > 0 else pd.Series(1.0, index=port.index)
    port_vt = (port * w).dropna()
    sd_vt = port_vt.std(ddof=1)
    vt_sharpe = float(port_vt.mean() / sd_vt * np.sqrt(BARS_PER_YEAR)) if sd_vt > 0 else 0.0
    t_vt = newey_west_t(port_vt.to_numpy())

    pos_frac = float(np.mean([s > 0 for s in wf_sharpes])) if wf_sharpes else 0.0
    return {
        "variant": variant["variant"],
        "lookback": variant["lookback"],
        "signal_smooth": variant["signal_smooth"],
        "allow_short": variant["allow_short"],
        "n_symbols": len(per_symbol),
        "mean_oos_wf": float(np.mean(wf_sharpes)) if wf_sharpes else 0.0,
        "symbols_pos_frac": pos_frac,
        "n_trades_oos": n_trades,
        "port_sharpe": port_sharpe,
        "t_newey_west": t_nw,
        "port_sharpe_vt": vt_sharpe,
        "t_newey_west_vt": t_vt,
        "notes": (f"smooth={variant['signal_smooth']}, short={variant['allow_short']}, n_sym={len(per_symbol)}"),
    }


def phase_variants(days: int = 2500) -> pd.DataFrame:
    print("=== Phase 1: variant sweep ===", flush=True)
    rows: list[dict] = []
    for i, var in enumerate(VARIANTS, 1):
        t0 = time.time()
        row = _portfolio_test(var, days=days)
        rows.append(row)
        print(
            f"[{i}/{len(VARIANTS)}] {var['variant']:<20s} "
            f"port_SR={row.get('port_sharpe', 0):+.3f} t_NW={row.get('t_newey_west', 0):+.2f} "
            f"vt_SR={row.get('port_sharpe_vt', 0):+.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    df = pd.DataFrame(rows)
    out_csv = OUT / "variants.csv"
    df.to_csv(out_csv, index=False)
    best = df.loc[df["port_sharpe"].idxmax()] if "port_sharpe" in df.columns and len(df) else None
    if best is not None and pd.notna(best.get("port_sharpe")):
        print(f"\nНайкращий варіант: {best['variant']} (port Sharpe {best['port_sharpe']:+.3f})", flush=True)
    print(f"Збережено: {out_csv}", flush=True)
    return df


def phase_audit(days: int = 1095, with_cscv: bool = True) -> pd.DataFrame:
    print("=== Phase 2: selective audit ts_momentum 1d ===", flush=True)
    from scalper_hft.validation.cell_audit import audit_cell

    csv_path = OUT / "audit_ts_momentum_1d.csv"
    jsonl_path = OUT / "audit_ts_momentum_1d.jsonl"
    done: set[tuple[str, str, str]] = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = {(str(r.strategy), str(r.symbol), str(r.interval)) for r in prev.itertuples(index=False)}

    rows: list[dict] = []
    for i, (strategy, symbol, interval) in enumerate(AUDIT_CELLS, 1):
        if (strategy, symbol, interval) in done:
            continue
        t0 = time.time()
        audit = audit_cell(strategy, symbol, interval, days, with_cscv=with_cscv)
        summary = audit.to_summary_dict()
        rows.append(summary)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(
            f"[{i}/{len(AUDIT_CELLS)}] {strategy} {symbol} {interval} "
            f"oos={summary.get('avg_oos_sharpe')} dsr={summary.get('dsr')} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(rows)


def phase_leaderboard() -> str:
    print("=== Phase 3: leaderboard ===", flush=True)
    from scalper_hft.research.leaderboard import render_leaderboard_markdown

    md = render_leaderboard_markdown(ROOT / "results")
    out_md = ROOT / "docs" / "reports" / "LEADERBOARD.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    csv_out = OUT / "leaderboard.csv"
    from scalper_hft.research.leaderboard import leaderboard_dataframe

    leaderboard_dataframe(ROOT / "results").to_csv(csv_out, index=False)
    print(f"Збережено: {out_md}", flush=True)
    print(f"Збережено: {csv_out}", flush=True)
    return str(out_md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["all", "variants", "audit", "leaderboard"],
        default="all",
    )
    ap.add_argument("--days", type=int, default=2500, help="днів для variant sweep (1d native)")
    ap.add_argument("--audit-days", type=int, default=1095, help="днів для cell audit")
    ap.add_argument("--no-cscv", action="store_true", help="швидкий аудит без CSCV/PBO")
    args = ap.parse_args()

    if args.phase in ("all", "variants"):
        phase_variants(days=args.days)
    if args.phase in ("all", "audit"):
        phase_audit(days=args.audit_days, with_cscv=not args.no_cscv)
    if args.phase in ("all", "leaderboard"):
        phase_leaderboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
