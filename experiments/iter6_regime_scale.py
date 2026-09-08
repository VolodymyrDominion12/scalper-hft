"""Iteration 6 — regime_scale overlay для pairs_arb: A/B на 3y + 400d.

Мотивація: iter4 показав, що pairs edge regime-залежний (LINK/BTC 3y +54.9%
але 400d −9.65%). regime_scale масштабує експозицію до 0.5 у high-vol/trend leg2
(BTC) — гіпотеза: зменшити втрати у несприятливих режимах при збереженні edge.

Запуск: uv run python experiments/iter6_regime_scale.py
Вихід: results/iter6_regime_scale.csv + таблиця в stdout.
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

# Пари з перевалідації iter5 + VALIDATED_PAIRS (lb=120 ядро)
PAIRS = [
    ("LINKUSDT", "BTCUSDT", 2.0, 0.3, 120),  # VALIDATED_PAIRS ядро
    ("LINKUSDT", "BTCUSDT", 2.0, 0.3, 240),  # iter5 альтернатива
    ("XRPUSDT", "BTCUSDT", 2.0, 0.3, 480),  # iter5
    ("LINKUSDT", "ETHUSDT", 2.0, 0.3, 240),  # iter5
    ("BTCUSDT", "ETHUSDT", 2.0, 0.3, 240),  # iter5
]

DAYS = 1095
TRAIN, TEST = 1500, 500
POSITION_PCT = 0.3


def load_leg(symbol: str) -> pd.DataFrame:
    from scalper_hft.data.access import ensure_klines

    return ensure_klines(symbol, "1h", DAYS, derive=False)


def run_variant(leg1: str, leg2: str, ez: float, xz: float, lb: int, regime_scale: bool) -> dict:
    d1, d2 = load_leg(leg1), load_leg(leg2)
    f1, f2 = download_funding(leg1, DAYS), download_funding(leg2, DAYS)
    strat = get_strategy("pairs_arb", entry_z=ez, exit_z=xz, lookback=lb, regime_scale=regime_scale)
    t0 = time.time()

    res = run_pairs_backtest(d1, d2, strat, f1, f2, position_pct=POSITION_PCT, maker_execution=True)
    m = res.metrics
    wf = run_pairs_walk_forward(
        d1, d2, strat, f1, f2, train_bars=TRAIN, test_bars=TEST, position_pct=POSITION_PCT, maker_execution=True
    )
    # останні 400 днів
    cut = d1.index[-1] - pd.Timedelta(days=400)
    d1c, d2c = d1[d1.index > cut], d2[d2.index > cut]
    resc = run_pairs_backtest(d1c, d2c, strat, f1, f2, position_pct=POSITION_PCT, maker_execution=True)
    mc = resc.metrics
    return {
        "pair": f"{leg1}/{leg2}",
        "params": f"z={ez},exit={xz},lb={lb}",
        "regime_scale": regime_scale,
        "n_wf_windows": wf["n_windows"],
        "wf_avg_oos_srh": round(wf["avg_oos_sharpe"], 5),
        "wf_pos_frac": round(wf["positive_windows"], 3),
        "bt_3y_ret": round(m.total_return, 4),
        "bt_3y_pf": round(m.profit_factor, 2),
        "bt_3y_trades": int(m.n_trades),
        "bt_3y_maxdd": round(m.max_drawdown, 4),
        "bt_3y_srh": round(m.sharpe_hourly, 5),
        "bt_400d_ret": round(mc.total_return, 4),
        "bt_400d_pf": round(mc.profit_factor, 2),
        "bt_400d_trades": int(mc.n_trades),
        "bt_400d_maxdd": round(mc.max_drawdown, 4),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    out_csv = Path("results/iter6_regime_scale.csv")
    rows: list[dict] = []
    for leg1, leg2, ez, xz, lb in PAIRS:
        for rs in (False, True):
            label = f"{leg1}/{leg2} lb={lb} regime_scale={rs}"
            try:
                row = run_variant(leg1, leg2, ez, xz, lb, rs)
                rows.append(row)
                print(row, flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"{label} ERROR: {exc}", flush=True)

    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {out_csv}")

        # Зведена A/B таблиця
        print("\n=== A/B: regime_scale=False vs True (3y ret / 400d ret / maxDD) ===")
        for leg1, leg2, ez, xz, lb in PAIRS:
            base = next(
                (
                    r
                    for r in rows
                    if r["pair"] == f"{leg1}/{leg2}"
                    and r["params"] == f"z={ez},exit={xz},lb={lb}"
                    and not r["regime_scale"]
                ),
                None,
            )
            scaled = next(
                (
                    r
                    for r in rows
                    if r["pair"] == f"{leg1}/{leg2}"
                    and r["params"] == f"z={ez},exit={xz},lb={lb}"
                    and r["regime_scale"]
                ),
                None,
            )
            if base and scaled:
                print(
                    f"{leg1}/{leg2} lb={lb}: "
                    f"3y {base['bt_3y_ret']:+.1%} -> {scaled['bt_3y_ret']:+.1%} | "
                    f"400d {base['bt_400d_ret']:+.1%} -> {scaled['bt_400d_ret']:+.1%} | "
                    f"maxDD {base['bt_3y_maxdd']:.1%} -> {scaled['bt_3y_maxdd']:.1%} | "
                    f"WFpos {base['wf_pos_frac']:.0%} -> {scaled['wf_pos_frac']:.0%}"
                )


if __name__ == "__main__":
    main()
