"""iter9 — momentum/trend на ДЕННОМУ горизонті з довгою історією (6.8 років).

Навіщо. Скрин на 3-річному 1m-кеші показав, що єдиний таймфрейм з невід'ємним
edge — 4h/1d (див. results/iter9/screen_all_1095d.csv), але 3 роки = лише 8 OOS
вікон по 100 барів у walk-forward і ~65 OOS-угод. Це замало, щоб відрізнити
edge від шуму. Денні бари не потребують 1m-кеша, тож завантажуємо НАТИВНІ 1d
klines (REST, до ~2500 барів = 6.8 року) і перевіряємо:

  1. чи тримається edge momentum/trend на горизонті ~7 років (а не 3);
  2. чи він період-залежний — окремо 2019–2022 (ведмеді/боковик) і 2023–2026
     (бичачий альт-ринок). Ітерація iter4 показала саме період-залежність
     supertrend 1d (+0.50 на 3y → −0.15 на 5y), і це треба перевірити на
     чистому денному вікні без 1m-обмеження;
  3. чи змінює висновок maker-виконання замість taker (інженерія витрат:
     round-trip 10 bps → 4 bps + slippage).

Запуск:
    .venv/bin/python experiments/iter9_daily_long_history.py [--days 2500]

Дані: спершу `python -m scalper_hft.cli download --symbol ... --interval 1d
--days 2500 --no-derive` (нативні денні klines; 1m-кеш не чіпається).
Результат: results/iter9/daily_long_history.csv + .md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.access import ensure_klines
from scalper_hft.data.store import get_store
from scalper_hft.strategies import get_strategy
from scalper_hft.validation.benchmark import buy_and_hold_sharpe
from scalper_hft.validation.deflated_sharpe import probabilistic_sharpe_ratio
from scalper_hft.validation.walk_forward import run_walk_forward

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter9"
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,"
    "AVAXUSDT,UNIUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,LTCUSDT,AAVEUSDT"
).split(",")

STRATEGIES = ("ts_momentum", "supertrend")
TRAIN, TEST = 250, 125

PERIODS = {
    "full": (None, None),
    "2019-2022": ("2019-01-01", "2023-01-01"),
    "2023-2026": ("2023-01-01", "2027-01-01"),
}


def _sharpe_tstat(oos_returns: pd.Series) -> tuple[float, float]:
    """Sharpe на бар і t-статистика середнього (per-bar, без ануалізації)."""
    r = oos_returns.dropna().to_numpy(dtype=float)
    if len(r) < 3 or r.std(ddof=1) == 0:
        return 0.0, 0.0
    sr = float(r.mean() / r.std(ddof=1))
    t = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
    return sr, t


def _cell(
    strategy_name: str,
    symbol: str,
    df: pd.DataFrame,
    funding: pd.DataFrame | None,
    *,
    is_maker: bool,
    period: str,
) -> dict:
    strategy = get_strategy(strategy_name)
    cost = CostModel.from_settings(get_settings(), df=df)
    wf = run_walk_forward(
        df,
        strategy,
        train_bars=TRAIN,
        test_bars=TEST,
        cost=cost,
        funding=funding,
        is_maker=is_maker,
        interval="1d",
        collect_oos_returns=True,
        purge_bars=2,
        embargo_bars=2,
        strict_data=False,
    )
    oos = wf.oos_returns if wf.oos_returns is not None else pd.Series(dtype=float)
    sr_bar, t_stat = _sharpe_tstat(oos)
    ann = float(np.sqrt(365.0))
    years = (df.index[-1] - df.index[0]).days / 365.25
    return {
        "strategy": strategy_name,
        "symbol": symbol,
        "execution": "maker" if is_maker else "taker",
        "period": period,
        "n_bars": len(df),
        "years": round(years, 2),
        "n_windows": len(wf.windows),
        "n_trades_oos": int(sum(w.n_trades for w in wf.windows)),
        "avg_is_sharpe": float(wf.avg_is_sharpe),
        "avg_oos_sharpe": float(wf.avg_oos_sharpe),
        "oos_pos_frac": float(wf.positive_windows_frac),
        "oos_sharpe_bar_ann": sr_bar * ann,
        "oos_t_stat": t_stat,
        "psr": probabilistic_sharpe_ratio(oos.to_numpy(dtype=float)) if len(oos) > 3 else 0.0,
        "benchmark_sharpe": buy_and_hold_sharpe(df),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2500)
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    store = get_store()
    rows: list[dict] = []
    for symbol in symbols:
        try:
            df_full = ensure_klines(symbol, "1d", args.days, derive=False)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {symbol}: {exc}")
            continue
        if df_full is None or df_full.empty:
            print(f"skip {symbol}: немає нативних 1d даних")
            continue
        try:
            funding = store.load_funding(symbol)
        except Exception:  # noqa: BLE001
            funding = None
        for strategy_name in STRATEGIES:
            for period, (lo, hi) in PERIODS.items():
                df = df_full
                if lo is not None:
                    df = df_full.loc[(df_full.index >= lo) & (df_full.index < hi)]
                if len(df) < TRAIN + TEST + 10:
                    continue
                for is_maker in (False, True):
                    t0 = time.time()
                    try:
                        row = _cell(strategy_name, symbol, df, funding, is_maker=is_maker, period=period)
                    except Exception as exc:  # noqa: BLE001
                        print(f"ERR {strategy_name} {symbol} {period} maker={is_maker}: {type(exc).__name__}: {exc}")
                        continue
                    rows.append(row)
                    print(
                        f"{strategy_name:<15s} {symbol:<9s} {period:<10s} "
                        f"{row['execution']:<5s} win={row['n_windows']:>3d} "
                        f"oos={row['avg_oos_sharpe']:+.3f} pos={row['oos_pos_frac']:.0%} "
                        f"t={row['oos_t_stat']:+.2f} tr={row['n_trades_oos']:>4d} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    pd.DataFrame(rows).to_csv(OUT / "daily_long_history.csv", index=False)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print("Немає результатів — спершу завантажте нативні 1d klines")
        return 1
    df_out.to_csv(OUT / "daily_long_history.csv", index=False)

    summary = (
        df_out.groupby(["strategy", "period", "execution"])
        .agg(
            n_symbols=("symbol", "nunique"),
            mean_oos=("avg_oos_sharpe", "mean"),
            med_oos=("avg_oos_sharpe", "median"),
            symbols_pos=("avg_oos_sharpe", lambda s: int((s > 0).sum())),
            mean_t=("oos_t_stat", "mean"),
            mean_trades=("n_trades_oos", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "daily_long_history_summary.csv", index=False)
    print("\n", summary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
