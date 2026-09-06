"""Iteration 1 — walk-forward порівняння «regime-сутності» vs одиночні стратегії.

Гіпотеза (дослідження: Kiploks, DaruFinance, Krishhiv + внутрішній OOS-sweep):
    мета-сутність, що ОБИРАЄ стратегію за режимом ринку (structure + vol),
    має кращий avg OOS Sharpe за одиночні стратегії на тих самих клітинках,
    а гістерезис (min_dwell_bars) і жорсткий вибір (best_prior) знижують
    churn/turnover без втрати OOS.

Порівнюємо на 1h (liquid alts), walk-forward з фіксованими параметрами:
    singles:      supertrend, cross_momentum, stoch_rsi
    supervisor:   regime_soft / contextual_hedge / best_prior
                  children = supertrend,cross_momentum,stoch_rsi
                  dwell ∈ {0, 3}

Запуск (з кореня репо):
    .venv/bin/python experiments/iter1_wf_compare.py [--days 365] [--workers 4]
Результат: results/iter1_wf_compare.csv + короткий markdown-звіт.
Без lookahead: walk_forward ріже вікна за часом; HMM фітиться на train-частині
вікна; параметри НЕ оптимізуються на IS (фіксовані дефолти) — це тест
гіпотези «зміна архітектури», не підгонка.
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.access import ensure_klines
from scalper_hft.strategies import get_strategy
from scalper_hft.validation.walk_forward import run_walk_forward

SYMBOLS = ["XRPUSDT", "AVAXUSDT", "UNIUSDT", "LINKUSDT", "AAVEUSDT", "ADAUSDT"]
INTERVAL = "1h"
BASE = "1m"
# Два набори дітей: офіційний дефолт (mean_reversion,supertrend,hmm_reversion) і
# «чистий» regime-спліт (тренд: supertrend/cross_momentum; рейндж: stoch_rsi).
CHILDREN_SETS = {
    "official": "mean_reversion,supertrend,hmm_reversion",
    "split": "supertrend,cross_momentum,stoch_rsi",
}


def _variants() -> list[tuple[str, dict]]:
    v: list[tuple[str, dict]] = [
        ("single:supertrend", {"name": "supertrend"}),
        ("single:stoch_rsi", {"name": "stoch_rsi"}),
    ]
    for cset, children in CHILDREN_SETS.items():
        for mode in ("regime_soft", "contextual_hedge", "best_prior"):
            for dwell in (0, 3):
                v.append(
                    (
                        f"sup:{cset}:{mode}:d{dwell}",
                        {
                            "name": "regime_supervisor",
                            "children": children,
                            "blend_mode": mode,
                            "min_dwell_bars": dwell,
                        },
                    )
                )
    # Ітерація 2: режимні гейти (vol_high_veto / trend_direction_gate) на
    # найбільш перспективних конфігураціях.
    for cset, children in CHILDREN_SETS.items():
        for mode in ("regime_soft", "best_prior"):
            for dwell in (0, 3):
                for tag, gates in (
                    ("vhv", {"vol_high_veto": True}),
                    ("tdg", {"trend_direction_gate": True}),
                    ("both", {"vol_high_veto": True, "trend_direction_gate": True}),
                ):
                    v.append(
                        (
                            f"sup:{cset}:{mode}:d{dwell}:{tag}",
                            {
                                "name": "regime_supervisor",
                                "children": children,
                                "blend_mode": mode,
                                "min_dwell_bars": dwell,
                                **gates,
                            },
                        )
                    )
    return v


VARIANTS: list[tuple[str, dict]] = _variants()


def _build_strategy(variant: dict):
    if variant["name"] == "regime_supervisor":
        return get_strategy(
            "regime_supervisor",
            strategies=variant["children"],
            blend_mode=variant["blend_mode"],
            min_dwell_bars=variant["min_dwell_bars"],
            n_hmm_states=3,
            hmm_fit_bars=2000,
            vol_high_veto=variant.get("vol_high_veto", False),
            trend_direction_gate=variant.get("trend_direction_gate", False),
        )
    return get_strategy(variant["name"])


def run_cell(symbol: str, variant_label: str, variant: dict, days: int, train: int, test: int, maker: bool) -> dict:
    settings = get_settings()
    strat = _build_strategy(variant)
    df = ensure_klines(symbol, INTERVAL, days, base_interval=BASE, derive=True, readonly=True)
    cost = CostModel(
        maker_fee=settings.maker_fee,
        taker_fee=settings.taker_fee,
        slippage_frac=settings.slippage_frac,
    )
    t0 = time.time()
    res = run_walk_forward(
        df,
        strat,
        train,
        test,
        cost=cost,
        position_pct=settings.position_pct,
        interval=INTERVAL,
        is_maker=maker,
    )
    row = {
        "symbol": symbol,
        "variant": variant_label,
        "kind": variant["name"],
        "blend_mode": str(variant.get("blend_mode", "")),
        "dwell": variant.get("min_dwell_bars", ""),
        "maker": maker,
        "n_bars": len(df),
        "n_windows": len(res.windows),
        "n_trades_oos": sum(w.n_trades for w in res.windows),
        "avg_is_sharpe": round(res.avg_is_sharpe, 4),
        "avg_oos_sharpe": round(res.avg_oos_sharpe, 4),
        "oos_positive_frac": round(res.positive_windows_frac, 4),
        "seconds": round(time.time() - t0, 1),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--symbols", type=str, default=",".join(SYMBOLS))
    ap.add_argument("--maker", action="store_true", default=True)
    args = ap.parse_args()

    symbols = [s for s in args.symbols.split(",") if s]
    out_csv = Path("results/iter1_wf_compare.csv")
    out_md = Path("results/iter1_wf_compare.md")

    cells = [
        (sym, label, variant, args.days, args.train, args.test, args.maker)
        for sym in symbols
        for label, variant in VARIANTS
    ]
    rows: list[dict] = []
    workers = args.workers if args.workers > 1 else 1
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_cell, *c) for c in cells]
            for f in futures:
                rows.append(f.result())
    else:
        for c in cells:
            rows.append(run_cell(*c))

    rows.sort(key=lambda r: (r["symbol"], r["variant"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # md-звіт: середнє по символах per variant
    from collections import defaultdict

    agg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        agg[r["variant"]].append(r)
    lines = ["# Iteration 1 — WF-порівняння regime-сутності vs singles (1h)", ""]
    lines.append(f"- days={args.days}, train={args.train}, test={args.test}, symbols={symbols}")
    children_desc = "; ".join(f"{k}={v}" for k, v in CHILDREN_SETS.items())
    lines.append(f"- children sets: {children_desc}; is_maker={args.maker}, без overlay")
    lines.append("")
    lines.append("| variant | avg OOS SR | pos win frac | n_trades OOS | per-symbol OOS SR |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(agg):
        rs = agg[label]
        mean_oos = sum(r["avg_oos_sharpe"] for r in rs) / len(rs)
        mean_pos = sum(r["oos_positive_frac"] for r in rs) / len(rs)
        trades = sum(r["n_trades_oos"] for r in rs)
        per_sym = " | ".join(f"{r['symbol']}:{r['avg_oos_sharpe']:+.3f}" for r in sorted(rs, key=lambda x: x["symbol"]))
        lines.append(f"| {label} | {mean_oos:+.4f} | {mean_pos:.0%} | {trades} | {per_sym} |")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nCSV: {out_csv}  MD: {out_md}")


if __name__ == "__main__":
    main()
