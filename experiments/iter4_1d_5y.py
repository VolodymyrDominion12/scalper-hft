"""Iteration 4 — 1d трендовий механізм на 5-річній історії (більше угод/вікон).

Передумова: `download --interval 1d --days 1900 --no-derive` для символів.
Гіпотези:
1) supertrend allow_short=True на 1d має стабільний OOS-плюс (3y: +0.50 avg,
   7/8 символів), але 5–12 угод/OOS-вікно — брак вибірки. 5y → ~2× угод, 7+ вікон.
2) atr_mult=2.0 (частіші фліпи) — гіпотеза більшої частоти без втрати edge.
3) 12h/8h компроміс уже відхилено (від'ємний) — не включаємо.

Запуск: .venv/bin/python experiments/iter4_1d_5y.py
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.access import ensure_klines
from scalper_hft.strategies import get_strategy
from scalper_hft.validation.walk_forward import run_walk_forward

SYMBOLS = ["AAVEUSDT", "AVAXUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"]
DAYS = 1900
TRAIN, TEST = 700, 150

VARIANTS: list[tuple[str, dict]] = [
    ("single:supertrend:ls", {"name": "supertrend", "allow_short": True}),
    ("single:supertrend:ls:mult2", {"name": "supertrend", "allow_short": True, "atr_mult": 2.0}),
    ("single:supertrend:lo", {"name": "supertrend", "allow_short": False}),
    ("single:cross_momentum", {"name": "cross_momentum"}),
]


def run_cell(symbol: str, label: str, variant: dict) -> dict:
    settings = get_settings()
    strat = get_strategy(variant["name"], **{k: v for k, v in variant.items() if k != "name"})
    df = ensure_klines(symbol, "1d", DAYS, derive=False)
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    t0 = time.time()
    res = run_walk_forward(
        df, strat, TRAIN, TEST, cost=cost, position_pct=settings.position_pct, interval="1d", is_maker=True
    )
    return {
        "symbol": symbol,
        "variant": label,
        "n_bars": len(df),
        "n_windows": len(res.windows),
        "n_trades_oos": sum(w.n_trades for w in res.windows),
        "avg_is_sharpe": round(res.avg_is_sharpe, 4),
        "avg_oos_sharpe": round(res.avg_oos_sharpe, 4),
        "oos_positive_frac": round(res.positive_windows_frac, 4),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    out_csv = Path("results/iter4_1d_5y.csv")
    out_md = Path("results/iter4_1d_5y.md")
    rows = []
    for sym in SYMBOLS:
        for label, variant in VARIANTS:
            rows.append(run_cell(sym, label, variant))
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
    lines = ["# Iteration 4 — 1d trend 5y WF (maker)", ""]
    lines.append(f"- days={DAYS}, train={TRAIN}, test={TEST}, symbols={SYMBOLS}")
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
