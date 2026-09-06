"""Iteration 3 — Walk-forward на довших горизонтах (4h/1d) з 3-річної бази.

Гіпотеза (research): тренд-фоловінг працює на горизонтах від тижнів до місяців
(Physica A: 3mo–1y; AQR TSMOM); на 1h напрямкового edge не підтвердилось
(iter1/2). Перевіряємо, чи 4h/1d maker дає OOS-позитивні клітинки, перш ніж
будувати MTF-сутність поверх.

Запуск: .venv/bin/python experiments/iter3_wf_mtf.py --interval 4h --days 1000
        (1d: --interval 1d --days 1000 --train 700 --test 120)
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.access import ensure_klines
from scalper_hft.strategies import get_strategy
from scalper_hft.validation.walk_forward import run_walk_forward

SYMBOLS = ["AAVEUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT", "AVAXUSDT", "UNIUSDT", "DOGEUSDT", "ETHUSDT"]

VARIANTS: list[tuple[str, dict]] = [
    ("single:supertrend:ls", {"name": "supertrend", "allow_short": True}),
    ("single:supertrend:lo", {"name": "supertrend", "allow_short": False}),
    ("single:cross_momentum", {"name": "cross_momentum"}),
    ("single:stoch_rsi", {"name": "stoch_rsi"}),
    ("single:smc_fvg", {"name": "smc_fvg"}),
    ("sup:split:regime_soft", {"name": "regime_supervisor", "children": "supertrend,cross_momentum,stoch_rsi", "blend_mode": "regime_soft", "min_dwell_bars": 2}),
    ("sup:split:best_prior", {"name": "regime_supervisor", "children": "supertrend,cross_momentum,stoch_rsi", "blend_mode": "best_prior", "min_dwell_bars": 2}),
]


def _build_strategy(variant: dict):
    if variant["name"] == "regime_supervisor":
        return get_strategy(
            "regime_supervisor",
            strategies=variant["children"],
            blend_mode=variant["blend_mode"],
            min_dwell_bars=variant["min_dwell_bars"],
            n_hmm_states=3,
            hmm_fit_bars=2000,
        )
    return get_strategy(variant["name"], **{k: v for k, v in variant.items() if k != "name"})


def run_cell(symbol: str, label: str, variant: dict, interval: str, days: int, train: int, test: int) -> dict:
    settings = get_settings()
    strat = _build_strategy(variant)
    df = ensure_klines(symbol, interval, days, base_interval="1m", derive=True, readonly=True)
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    t0 = time.time()
    res = run_walk_forward(
        df, strat, train, test, cost=cost, position_pct=settings.position_pct, interval=interval, is_maker=True
    )
    return {
        "symbol": symbol,
        "variant": label,
        "interval": interval,
        "n_bars": len(df),
        "n_windows": len(res.windows),
        "n_trades_oos": sum(w.n_trades for w in res.windows),
        "avg_is_sharpe": round(res.avg_is_sharpe, 4),
        "avg_oos_sharpe": round(res.avg_oos_sharpe, 4),
        "oos_positive_frac": round(res.positive_windows_frac, 4),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=str, default="4h")
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--symbols", type=str, default=",".join(SYMBOLS))
    args = ap.parse_args()

    symbols = [s for s in args.symbols.split(",") if s]
    out_csv = Path(f"results/iter3_{args.interval}.csv")
    out_md = Path(f"results/iter3_{args.interval}.md")

    rows = []
    for sym in symbols:
        for label, variant in VARIANTS:
            rows.append(run_cell(sym, label, variant, args.interval, args.days, args.train, args.test))
            print(f"done {sym} {label}", flush=True)

    rows.sort(key=lambda r: (r["variant"], r["symbol"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        agg[r["variant"]].append(r)
    lines = [f"# Iteration 3 — WF {args.interval} maker (3y база)", ""]
    lines.append(f"- days={args.days}, train={args.train}, test={args.test}, symbols={symbols}")
    lines.append("")
    lines.append("| variant | avg OOS SR | pos frac | trades | per-symbol OOS |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(agg):
        rs = agg[label]
        mo = sum(r["avg_oos_sharpe"] for r in rs) / len(rs)
        mp = sum(r["oos_positive_frac"] for r in rs) / len(rs)
        tr = sum(r["n_trades_oos"] for r in rs)
        per = " | ".join(f"{r['symbol']}:{r['avg_oos_sharpe']:+.2f}" for r in sorted(rs, key=lambda x: x["symbol"]))
        lines.append(f"| {label} | {mo:+.4f} | {mp:.0%} | {tr} | {per} |")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nCSV: {out_csv}  MD: {out_md}")


if __name__ == "__main__":
    main()
