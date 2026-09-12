"""iter9 — портфельний тест daily momentum: чи дає диверсифікація значущий edge.

Питання. На 6-річній нативній 1d-історії `ts_momentum` дає середній OOS Sharpe
+0.13 на 15 символах (12 із 15 позитивних, 2019–2022 +0.16 / 2023–2026 +0.02) —
тобто edge схожий на справжній, але занадто малий, щоб пройти гейт 0.3 на
ОДНОМУ символі. Класичний результат TSMOM-літератури: диверсифікований
портфель дає значно вищий Sharpe, ніж середній символ, бо ідіосинкратичний шум
гаситься кореляцією. Цей скрипт перевіряє саме це — і додає коректну
статистику (Newey–West t з поправкою на автокореляцію + stationary bootstrap).

Що рахуємо:
    1. WF OOS-дохідності кожного символу (train=250/test=125, maker і taker);
    2. рівноважний портфель (mean по символах на кожному барі);
    3. Sharpe портфеля + t-стат (Newey–West, лаги = 20) + bootstrap CI;
    4. середню парну кореляцію OOS-дохідностей (чи справді диверсифікація є);
    5. бенчмарк: рівноважний buy&hold тих самих символів за той самий період.

Запуск: .venv/bin/python experiments/iter9_ts_momentum_portfolio.py
Результат: results/iter9/ts_momentum_portfolio.md (+ .csv по символах)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scalper_hft.backtest.execution import CostModel
from scalper_hft.config import get_settings
from scalper_hft.data.access import ensure_klines
from scalper_hft.strategies import get_strategy
from scalper_hft.validation.walk_forward import run_walk_forward

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter9"
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,"
    "AVAXUSDT,UNIUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,LTCUSDT,AAVEUSDT"
).split(",")
TRAIN, TEST = 250, 125
BARS_PER_YEAR = 365.0
NW_LAGS = 20


def newey_west_t(returns: np.ndarray, lags: int = NW_LAGS) -> float:
    """t-статистика середнього з HAC-поправкою (Newey–West, Bartlett)."""
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


def bootstrap_sharpe_ci(returns: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Stationary bootstrap (Politis–Romano, середня довжина блоку 10) → 95% CI Sharpe."""
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
        out[b] = s.mean() / sd * np.sqrt(BARS_PER_YEAR) if sd > 0 else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2500)
    ap.add_argument("--strategy", default="ts_momentum")
    ap.add_argument("--lookback", type=int, default=20, help="lookback стратегії (тест плато)")
    ap.add_argument("--out-suffix", default="", help="суфікс імені артефактів (для серії запусків)")
    ap.add_argument("--vol-window", type=int, default=60, help="вікно vol-targeting overlay (днів)")
    args = ap.parse_args()

    strategy = get_strategy(args.strategy, lookback=args.lookback)
    cost = CostModel.from_settings(get_settings())

    per_symbol: dict[str, pd.Series] = {}
    bh: dict[str, pd.Series] = {}
    rows: list[dict] = []
    for symbol in SYMBOLS:
        try:
            df = ensure_klines(symbol, "1d", args.days, derive=False)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {symbol}: {exc}")
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
        per_symbol[symbol] = oos
        bh[symbol] = df["close"].pct_change().reindex(oos.index).fillna(0.0)
        sd = oos.std(ddof=1)
        rows.append(
            {
                "symbol": symbol,
                "n_bars_oos": len(oos),
                "n_windows": len(wf.windows),
                "oos_sharpe_ann": float(oos.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0,
                "avg_oos_sharpe_wf": float(wf.avg_oos_sharpe),
                "oos_pos_frac": float(wf.positive_windows_frac),
                "n_trades_oos": int(sum(w.n_trades for w in wf.windows)),
            }
        )
        print(f"{symbol:<9s} bars={len(oos):>4d} wf_sr={wf.avg_oos_sharpe:+.3f} pos={wf.positive_windows_frac:.0%}")

    if not per_symbol:
        print("Немає даних")
        return 1

    mat = pd.DataFrame(per_symbol)
    port = mat.mean(axis=1).dropna()
    bh_mat = pd.DataFrame(bh)
    bh_port = bh_mat.mean(axis=1).reindex(port.index).dropna()

    sd = port.std(ddof=1)
    port_sharpe = float(port.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    t_nw = newey_west_t(port.to_numpy())
    t_naive = float(port.mean() / (port.std(ddof=1) / np.sqrt(len(port)))) if port.std(ddof=1) > 0 else 0.0
    lo, hi = bootstrap_sharpe_ci(port.to_numpy())
    corr = mat.corr()
    avg_corr = float(corr.to_numpy()[np.triu_indices_from(corr.to_numpy(), k=1)].mean())
    sd_bh = bh_port.std(ddof=1)
    bh_sharpe = float(bh_port.mean() / sd_bh * np.sqrt(BARS_PER_YEAR)) if sd_bh > 0 else 0.0

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / f"ts_momentum_portfolio_symbols{args.out_suffix}.csv", index=False)

    # ── Overlay: vol-targeting на рівні портфеля (risk management, не альфа) ──
    # Ідея (Narang гл. 10; Moreira–Muir volatility-managed portfolios): та сама
    # альфа, але експозиція масштабується обернено до реалізованої волатильності
    # портфеля (rolling 60d, shift(1) — без lookahead), ціль = медіанна
    # волатильність (середнє плече ≈ 1). Перевіряємо, чи піднімає це Sharpe.
    vol_win = int(args.vol_window)
    realized = port.rolling(vol_win, min_periods=20).std(ddof=0).shift(1)
    med_vol = float(realized.median())
    if med_vol > 0:
        w = (med_vol / realized).clip(0.0, 2.0).fillna(0.0)
    else:
        w = pd.Series(1.0, index=port.index)
    port_vt = (port * w).dropna()
    sd_vt = port_vt.std(ddof=1)
    vt_sharpe = float(port_vt.mean() / sd_vt * np.sqrt(BARS_PER_YEAR)) if sd_vt > 0 else 0.0
    lo_vt, hi_vt = bootstrap_sharpe_ci(port_vt.to_numpy())
    t_vt = newey_west_t(port_vt.to_numpy())

    def max_dd(series: pd.Series) -> float:
        eq = (1.0 + series).cumprod()
        return float((eq / eq.cummax() - 1.0).min())

    overlay_block = [
        "## Overlay: vol-targeting портфеля (та сама альфа, інший ризик)",
        "",
        f"- Базова: Sharpe {port_sharpe:+.3f}, maxDD {max_dd(port):.1%}, t_NW {t_nw:+.2f}",
        f"- Vol-targeted (вікно {vol_win}d, ціль = медіанна σ, плече ≤2): "
        f"Sharpe {vt_sharpe:+.3f}, maxDD {max_dd(port_vt):.1%}, t_NW {t_vt:+.2f}, "
        f"bootstrap 95% CI [{lo_vt:+.2f}, {hi_vt:+.2f}]",
        f"- Δ Sharpe = {vt_sharpe - port_sharpe:+.3f}",
        "",
    ]

    lines = [
        f"# Портфельний тест daily momentum (`{args.strategy}` lb={args.lookback}, maker, 1d, {args.days} днів)",
        "",
        f"- Символів: {len(per_symbol)} · OOS-барів у портфелі: {len(port)} "
        f"({port.index[0]:%Y-%m-%d} → {port.index[-1]:%Y-%m-%d})",
        f"- Середній OOS Sharpe по символах (WF-вікна): {df_rows['avg_oos_sharpe_wf'].mean():+.3f} "
        f"(медіана {df_rows['avg_oos_sharpe_wf'].median():+.3f}, позитивних "
        f"{int((df_rows['avg_oos_sharpe_wf'] > 0).sum())}/{len(df_rows)})",
        f"- Рівноважний портфель: **Sharpe {port_sharpe:+.3f}** (річний), "
        f"t_naive={t_naive:+.2f}, **t_Newey–West(20)={t_nw:+.2f}**, "
        f"bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]",
        f"- Середня парна кореляція OOS-дохідностей: {avg_corr:+.3f}",
        f"- Бенчмарк (рівноважний buy&hold тих самих символів): Sharpe {bh_sharpe:+.3f}",
        "",
        *overlay_block,
        "## По символах",
        "",
        df_rows.round(3).to_markdown(index=False),
        "",
        "## Кореляційна матриця OOS-дохідностей",
        "",
        corr.round(2).to_markdown(),
        "",
    ]
    report = "\n".join(lines)
    (OUT / f"ts_momentum_portfolio{args.out_suffix}.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
