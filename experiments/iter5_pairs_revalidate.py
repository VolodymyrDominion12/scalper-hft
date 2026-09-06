"""Iteration 5 — перевалідація pairs_arb на поточному коді (3y, 1h, maker).

Мотивація: документована валідація (2026-08-30) не відтворюється після фіксів
моделі виконання (XRP/BTC +15.9% -> +5.2% на тому ж вікні). Перераховуємо на
3-річній 1h-базі з багатьма WF-вікнами.

Запуск: .venv/bin/python experiments/iter5_pairs_revalidate.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd

from scalper_hft.backtest.pairs import run_pairs_backtest, run_pairs_walk_forward
from scalper_hft.config import get_settings
from scalper_hft.data.downloader import download_funding
from scalper_hft.strategies import get_strategy

settings = get_settings()

PAIRS = [
    ("XRPUSDT", "BTCUSDT", 2.0, 0.3, 480),
    ("LINKUSDT", "BTCUSDT", 2.0, 0.3, 240),
    ("LINKUSDT", "ETHUSDT", 2.0, 0.3, 240),
    ("BTCUSDT", "ETHUSDT", 2.0, 0.3, 240),
]

DAYS = 1095
TRAIN, TEST = 1500, 500


def load_leg(symbol: str) -> pd.DataFrame:
    from scalper_hft.data.access import ensure_klines

    return ensure_klines(symbol, "1h", DAYS, derive=False)


def main() -> None:
    out_csv = Path("results/iter5_pairs.csv")
    rows = []
    for leg1, leg2, ez, xz, lb in PAIRS:
        try:
            d1, d2 = load_leg(leg1), load_leg(leg2)
            f1, f2 = download_funding(leg1, DAYS), download_funding(leg2, DAYS)
            strat = get_strategy("pairs_arb", entry_z=ez, exit_z=xz, lookback=lb)
            t0 = time.time()
            res = run_pairs_backtest(d1, d2, strat, f1, f2, position_pct=0.3, maker_execution=True)
            m = res.metrics
            wf = run_pairs_walk_forward(
                d1, d2, strat, f1, f2, train_bars=TRAIN, test_bars=TEST, position_pct=0.3, maker_execution=True
            )
            # останні 400 днів
            cut = d1.index[-1] - pd.Timedelta(days=400)
            d1c, d2c = d1[d1.index > cut], d2[d2.index > cut]
            resc = run_pairs_backtest(d1c, d2c, strat, f1, f2, position_pct=0.3, maker_execution=True)
            mc = resc.metrics
            row = {
                "pair": f"{leg1}/{leg2}",
                "params": f"z={ez},exit={xz},lb={lb}",
                "n_bars": len(d1),
                "n_wf_windows": wf["n_windows"],
                "wf_avg_oos_srh": round(wf["avg_oos_sharpe"], 5),
                "wf_pos_frac": round(wf["positive_windows"], 3),
                "bt_3y_ret": round(m.total_return, 4),
                "bt_3y_pf": round(m.profit_factor, 2),
                "bt_3y_trades": int(m.n_trades),
                "bt_3y_srh": round(m.sharpe_hourly, 5),
                "bt_400d_ret": round(mc.total_return, 4),
                "bt_400d_pf": round(mc.profit_factor, 2),
                "bt_400d_trades": int(mc.n_trades),
                "seconds": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(row, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{leg1}/{leg2} ERROR: {exc}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
