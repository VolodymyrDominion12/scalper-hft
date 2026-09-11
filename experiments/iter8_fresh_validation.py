"""iter8: fresh-data validation regime-selector (символи поза iter7 burn)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FRESH_SYMBOLS = ("ATOMUSDT", "NEARUSDT", "DOTUSDT")
SLEEVE_STRATEGIES = ("supertrend", "funding_carry", "smc_fvg")
MIN_TRADES = 30


def _wf_oos_sharpe(
    df: pd.DataFrame,
    strategy_name: str,
    *,
    symbol: str,
    days: int,
    train: int = 500,
    test: int = 200,
) -> tuple[float, int]:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.downloader import download_agg_trades, download_funding
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.walk_forward import run_walk_forward

    settings = get_settings()
    cost = CostModel.from_settings(settings, df=df)
    strat = get_strategy(strategy_name)
    trades = download_agg_trades(symbol, days) if strat.needs_trades else None
    funding = download_funding(symbol, days) if strat.needs_funding else None
    wf = run_walk_forward(
        df,
        strat,
        train,
        test,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
    )
    n_trades = sum(w.n_trades for w in wf.windows)
    return float(wf.avg_oos_sharpe), int(n_trades)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--symbols", default=",".join(FRESH_SYMBOLS))
    args = parser.parse_args()

    from scalper_hft.data.access import ensure_klines
    from scalper_hft.validation.benchmark import buy_and_hold_sharpe
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    rows: list[dict[str, object]] = []
    oos_returns: list[float] = []

    for sym in symbols:
        df = ensure_klines(sym, args.interval, args.days)
        if df is None or df.empty:
            continue
        bench = buy_and_hold_sharpe(df)
        for strat in SLEEVE_STRATEGIES:
            sr, n_tr = _wf_oos_sharpe(df, strat, symbol=sym, days=args.days)
            rows.append(
                {
                    "symbol": sym,
                    "strategy": strat,
                    "oos_sharpe": sr,
                    "n_trades": n_tr,
                    "benchmark_sharpe": bench,
                    "degenerate": n_tr < MIN_TRADES,
                }
            )
            if n_tr >= MIN_TRADES:
                oos_returns.append(sr)

    out = Path("results/iter8_fresh_validation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Записано {out} ({len(rows)} рядків)")

    if len(oos_returns) >= 3:
        dsr = deflated_sharpe_ratio(np.array(oos_returns, dtype=float), n_trials=len(rows))
        print(f"Portfolio DSR (n={len(oos_returns)} cells): {dsr:.3f}")
        print("PASS" if dsr > 0.95 else "FAIL (потрібен DSR>0.95)")


if __name__ == "__main__":
    main()
