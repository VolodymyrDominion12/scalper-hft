"""iter13 — ts_momentum на тижневому таймфреймі (1w), two-stage protocol.

Pre-registration: docs/reports/hypothesis_iter13.md (ЗАФІКСОВАНО до прогону).
Запуск: uv run python experiments/iter13_weekly_momentum.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter13"
OUT.mkdir(parents=True, exist_ok=True)

CORE_15 = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,"
    "AVAXUSDT,UNIUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,LTCUSDT,AAVEUSDT"
).split(",")

# Зафіксовано у hypothesis_iter13.md — не змінювати після прогону.
TRAIN_W, TEST_W = 50, 30  # test=30 — мінімум рушія run_backtest (>=30 барів)
BARS_PER_YEAR_1W = 52.0
DAYS = 2600
NW_LAGS = 8
LOOKBACKS = (4, 8, 13, 26)  # ≈ 1/2/3/6 місяців
SEL_START, SEL_END = "2019-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2026-12-31"


def newey_west_t(returns: np.ndarray, lags: int = NW_LAGS) -> float:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3:
        return 0.0
    mu = float(r.mean())
    dev = r - mu
    var = float((dev @ dev) / n)
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float((dev[lag:] @ dev[:-lag]) / n)
    if var <= 0:
        return 0.0
    return float(mu / np.sqrt(var / n))


def sharpe(returns: np.ndarray, bpy: float = BARS_PER_YEAR_1W) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 3:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0


def max_dd(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float(((eq / peak) - 1.0).min())


def bootstrap_ci(
    returns: np.ndarray, bpy: float = BARS_PER_YEAR_1W, n_boot: int = 2000, seed: int = 42
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


def phase_wf_returns(allow_short: bool) -> dict[int, pd.DataFrame]:
    """WF per-symbol OOS для кожного lookback зі сітки (альфа інакше заморожена)."""
    tag = "ls" if allow_short else "long"
    print(f"=== iter13 фаза 1: ts_momentum 1w CORE_15 per-symbol OOS ({tag}) ===", flush=True)
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.trial_ledger import default_path, record_trial
    from scalper_hft.validation.walk_forward import run_walk_forward

    t0 = time.time()
    cost = CostModel.from_settings(get_settings())
    mats: dict[int, pd.DataFrame] = {}
    for lookback in LOOKBACKS:
        strategy = get_strategy("ts_momentum", lookback=lookback, signal_smooth=1, allow_short=allow_short)
        per_symbol: dict[str, pd.Series] = {}
        skipped: list[str] = []
        for symbol in CORE_15:
            try:
                df = ensure_klines(symbol, "1w", DAYS, derive=False)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{symbol}: {type(exc).__name__}")
                continue
            if df is None or len(df) < TRAIN_W + TEST_W + 10:
                skipped.append(f"{symbol}: bars={0 if df is None else len(df)}")
                continue
            wf = run_walk_forward(
                df,
                strategy,
                train_bars=TRAIN_W,
                test_bars=TEST_W,
                cost=cost,
                is_maker=True,
                interval="1w",
                collect_oos_returns=True,
                purge_bars=1,
                embargo_bars=1,
                strict_data=False,
            )
            oos = wf.oos_returns.dropna()
            if oos.empty:
                skipped.append(f"{symbol}: empty OOS")
                continue
            per_symbol[symbol] = oos
        if skipped:
            print(f"    lb={lookback} skip: {skipped}", flush=True)
        mat = pd.DataFrame(per_symbol)
        mats[lookback] = mat
        mat.to_parquet(OUT / f"ts_oos_matrix_1w_lb{lookback}_{tag}.parquet")
        record_trial(
            default_path(),
            strategy="ts_momentum",
            symbol="PORTFOLIO_CORE15",
            purpose="iter13:weekly/1w",
            n_trials=1,
            extra={"lookback": lookback, "allow_short": allow_short, "n_symbols": mat.shape[1]},
        )
        print(f"    lb={lookback}: матриця {mat.shape} ({time.time() - t0:.0f}s)", flush=True)
    return mats


def phase_two_stage(mats: dict[int, pd.DataFrame]) -> dict:
    """H13-A: selection half обирає lookback, validation half — один прогін."""
    print("=== iter13 фаза 2 (H13-A): two-stage selection ===", flush=True)
    sel_rows = []
    ports: dict[int, pd.Series] = {}
    for lb, mat in mats.items():
        port = mat.mean(axis=1).dropna()
        ports[lb] = port
        sel = port.loc[SEL_START:SEL_END]
        sel_rows.append(
            {
                "lookback": lb,
                "sel_sharpe": sharpe(sel.to_numpy()),
                "sel_tnw": newey_west_t(sel.to_numpy()),
                "sel_weeks": int(len(sel)),
            }
        )
    sel_df = pd.DataFrame(sel_rows)
    chosen = int(sel_df.loc[sel_df.sel_sharpe.idxmax(), "lookback"])
    print(f"    selection: {sel_rows} | обрано lookback={chosen}", flush=True)

    val = ports[chosen].loc[VAL_START:VAL_END]
    s_val = sharpe(val.to_numpy())
    t_val = newey_west_t(val.to_numpy())
    ci_lo, ci_hi = bootstrap_ci(val.to_numpy())
    dd_val = max_dd(val.to_numpy())
    promote = bool(s_val > 0 and t_val >= 2.0 and ci_lo > 0)
    print(
        f"    validation: SR={s_val:+.3f} t_NW={t_val:+.2f} CI=[{ci_lo:+.2f},{ci_hi:+.2f}] "
        f"maxDD={dd_val:+.2%} weeks={len(val)} | promote={promote}",
        flush=True,
    )
    return {
        "hypothesis": "H13-A",
        "chosen_lookback": chosen,
        "sel_rows": sel_rows,
        "val_sharpe": s_val,
        "val_tnw": t_val,
        "val_ci_lo": ci_lo,
        "val_ci_hi": ci_hi,
        "val_maxdd": dd_val,
        "val_weeks": int(len(val)),
        "full_sharpe": sharpe(ports[chosen].to_numpy()),
        "full_tnw": newey_west_t(ports[chosen].to_numpy()),
        "promote_to_monitoring": promote,
        "decision_rule": "validation Sharpe > 0 AND t_NW >= 2.0 AND bootstrap CI excludes 0",
    }


def phase_control_ls(chosen_lookback: int) -> dict:
    """Контроль: long-short варіант для ОБРАНОГО lookback (очікування: слабший)."""
    print("=== iter13 фаза 3 (контроль): long-short ===", flush=True)
    mats_ls = phase_wf_returns(allow_short=True)
    mat = mats_ls.get(chosen_lookback)
    if mat is None or mat.empty:
        return {"error": "no ls matrix"}
    port = mat.mean(axis=1).dropna()
    val = port.loc[VAL_START:VAL_END]
    out = {
        "lookback": chosen_lookback,
        "val_sharpe_ls": sharpe(val.to_numpy()),
        "val_tnw_ls": newey_west_t(val.to_numpy()),
        "full_sharpe_ls": sharpe(port.to_numpy()),
        "full_tnw_ls": newey_west_t(port.to_numpy()),
    }
    print(f"    LS validation: SR={out['val_sharpe_ls']:+.3f} t_NW={out['val_tnw_ls']:+.2f}", flush=True)
    return out


def phase_value_added(mat_1w: pd.DataFrame, chosen_lookback: int) -> dict:
    """H13-B: 50/50 портфель 1w ⊕ 1d (матриця iter12) на спільному OOS-span."""
    print("=== iter13 фаза 4 (H13-B): value-added 1w ⊕ 1d ===", flush=True)
    path_1d = ROOT / "results" / "iter12" / "ts_oos_matrix_1d.parquet"
    if not path_1d.exists():
        return {"hypothesis": "H13-B", "error": "немає матриці iter12 1d"}
    port_1w = mat_1w.mean(axis=1).dropna()
    port_1d = pd.read_parquet(path_1d).mean(axis=1).dropna()
    # 1w → денна сітка: тижнева дохідність розподіляється на тиждень (ffill equity).
    eq_1w = (1.0 + port_1w).cumprod()
    eq_1d = (1.0 + port_1d).cumprod()
    idx = eq_1d.index
    eq_1w_daily = eq_1w.reindex(idx, method="ffill").dropna()
    eq_1d_al = eq_1d.loc[eq_1w_daily.index]
    r_1w = eq_1w_daily.pct_change().dropna()
    r_1d = eq_1d_al.pct_change().dropna()
    common = r_1w.index.intersection(r_1d.index)
    if len(common) < 100:
        return {"hypothesis": "H13-B", "error": f"overlap too short: {len(common)}"}
    w, d = r_1w.loc[common], r_1d.loc[common]
    combined = 0.5 * w + 0.5 * d
    bpy = 365.0
    out = {
        "hypothesis": "H13-B",
        "overlap_days": int(len(common)),
        "w_sharpe": sharpe(w.to_numpy(), bpy),
        "d_sharpe": sharpe(d.to_numpy(), bpy),
        "combined_sharpe": sharpe(combined.to_numpy(), bpy),
        "w_maxdd": max_dd(w.to_numpy()),
        "d_maxdd": max_dd(d.to_numpy()),
        "combined_maxdd": max_dd(combined.to_numpy()),
        "w_d_corr": float(w.corr(d)),
    }
    out["diversification_adds_value"] = bool(
        out["combined_sharpe"] > max(out["w_sharpe"], out["d_sharpe"])
        and out["combined_maxdd"] > max(out["w_maxdd"], out["d_maxdd"])
    )
    print(
        f"    overlap={out['overlap_days']}d corr={out['w_d_corr']:+.2f} | "
        f"1w SR={out['w_sharpe']:+.2f} | 1d SR={out['d_sharpe']:+.2f} | "
        f"combined SR={out['combined_sharpe']:+.2f} | adds={out['diversification_adds_value']}",
        flush=True,
    )
    return out


def write_report(ts: dict, ls: dict, va: dict, mats: dict[int, pd.DataFrame]) -> str:
    print("=== iter13 фаза 5: звіт ===", flush=True)
    chosen = ts["chosen_lookback"]
    md: list[str] = []
    md.append("# iter13 — ts_momentum 1w (тижневий momentum), two-stage protocol\n")
    md.append("Дата: 2026-09-12 · Pre-registration: [hypothesis_iter13.md](hypothesis_iter13.md)")
    md.append("Дані: LIVE `binanceusdm`, нативні 1w klines CORE_15 (перше використання 1w у проєкті).\n")
    md.append("## TL;DR\n")
    md.append("| Гіпотеза | Результат | Деталі |")
    md.append("|---|---|---|")
    verdict = "PASS → monitoring" if ts["promote_to_monitoring"] else "FAIL"
    md.append(
        f"| H13-A ts_momentum 1w long-only CORE_15 | {verdict} | "
        f"lookback={chosen}w; validation SR {ts['val_sharpe']:+.3f}, t_NW {ts['val_tnw']:+.2f}, "
        f"CI [{ts['val_ci_lo']:+.2f}, {ts['val_ci_hi']:+.2f}] |"
    )
    md.append(
        f"| контроль long-short | діагностика | validation SR {ls.get('val_sharpe_ls', float('nan')):+.3f}, "
        f"t_NW {ls.get('val_tnw_ls', float('nan')):+.2f} |"
    )
    if "error" not in va:
        md.append(
            f"| H13-B value-added 1w⊕1d | {'PASS' if va['diversification_adds_value'] else 'FAIL'} | "
            f"corr={va['w_d_corr']:+.2f}; combined SR {va['combined_sharpe']:+.2f} |"
        )
    md.append("")
    md.append("## H13-A: selection half (2019–2022)\n")
    md.append("| lookback (тижні) | selection Sharpe | t_NW | тижнів |")
    md.append("|---:|---:|---:|---:|")
    for r in ts["sel_rows"]:
        md.append(f"| {r['lookback']} | {r['sel_sharpe']:+.3f} | {r['sel_tnw']:+.2f} | {r['sel_weeks']} |")
    md.append(f"\nОбрано lookback = **{chosen}** тижнів.\n")
    md.append("## H13-A: validation half (2023–2026, невидима під час selection)\n")
    md.append("| Метрика | Значення |")
    md.append("|---|---:|")
    md.append(f"| Sharpe (bpy=52) | {ts['val_sharpe']:+.3f} |")
    md.append(f"| t_Newey–West(8) | {ts['val_tnw']:+.2f} |")
    md.append(f"| bootstrap 95% CI | [{ts['val_ci_lo']:+.2f}, {ts['val_ci_hi']:+.2f}] |")
    md.append(f"| maxDD | {ts['val_maxdd']:+.2%} |")
    md.append(f"| OOS-тижнів | {ts['val_weeks']} |")
    md.append(f"| (довідково, повний OOS) Sharpe / t_NW | {ts['full_sharpe']:+.3f} / {ts['full_tnw']:+.2f} |")
    md.append(
        f"\nРішення (заздалегідь): validation Sharpe>0 AND t_NW≥2.0 AND CI без 0 → "
        f"**{'PROMOTE → monitoring' if ts['promote_to_monitoring'] else 'НЕ промотувати'}**.\n"
    )
    md.append("## Контроль long-short (той самий lookback)\n")
    md.append("| Варіант | validation Sharpe | t_NW | повний OOS Sharpe | t_NW |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(
        f"| long-only | {ts['val_sharpe']:+.3f} | {ts['val_tnw']:+.2f} | {ts['full_sharpe']:+.3f} | {ts['full_tnw']:+.2f} |"
    )
    md.append(
        f"| long-short | {ls.get('val_sharpe_ls', float('nan')):+.3f} | {ls.get('val_tnw_ls', float('nan')):+.2f} | "
        f"{ls.get('full_sharpe_ls', float('nan')):+.3f} | {ls.get('full_tnw_ls', float('nan')):+.2f} |"
    )
    md.append("")
    if "error" not in va:
        md.append("## H13-B: value-added 1w ⊕ 1d\n")
        md.append(f"Спільний span: {va['overlap_days']} днів; кореляція 1w↔1d = **{va['w_d_corr']:+.2f}**.\n")
        md.append("| Портфель | Sharpe | maxDD |")
        md.append("|---|---:|---:|")
        md.append(f"| ts_momentum 1w CORE_15 | {va['w_sharpe']:+.2f} | {va['w_maxdd']:+.2%} |")
        md.append(f"| ts_momentum 1d CORE_15 | {va['d_sharpe']:+.2f} | {va['d_maxdd']:+.2%} |")
        md.append(f"| combined 50/50 | {va['combined_sharpe']:+.2f} | {va['combined_maxdd']:+.2%} |")
        md.append(
            f"\nВердикт: **{'диверсифікація додає вартість' if va['diversification_adds_value'] else 'без доданої вартості'}** "
            "(portfolio-construction evidence, не paper-gate).\n"
        )
    md.append("## Per-symbol OOS Sharpe (обраний lookback, повний OOS)\n")
    mat = mats[chosen]
    rows = sorted(
        ((sym, sharpe(mat[sym].dropna().to_numpy())) for sym in mat.columns),
        key=lambda x: x[1],
        reverse=True,
    )
    md.append("| symbol | Sharpe |")
    md.append("|---|---:|")
    for sym, s in rows:
        md.append(f"| {sym} | {s:+.2f} |")
    md.append("\n## Вердикт для paper\n")
    if ts["promote_to_monitoring"]:
        md.append(
            f"- `ts_momentum 1w` long-only CORE_15 (lookback={chosen}w) → **monitoring** "
            "поруч із pairs_arb LINK/BTC і ts_momentum 1d/4h."
        )
    else:
        md.append(
            "- `ts_momentum 1w` **не промотується** (гейт H13-A не пройдено) — "
            "monitoring-набір iter11 лишається без змін."
        )
    md.append("\n⚠ 1w-вікно після цього прогону спалене: див. `docs/reports/oos_usage.md`.")
    out_md = ROOT / "docs" / "reports" / "iter13_weekly_momentum.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    (OUT / "results.json").write_text(
        json.dumps({"ts": ts, "ls": ls, "va": va}, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
    print(f"Звіт: {out_md}", flush=True)
    return str(out_md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "wf", "report"], default="all")
    args = ap.parse_args()
    if args.phase in ("all", "wf"):
        mats = phase_wf_returns(allow_short=False)
    else:
        mats = {lb: pd.read_parquet(OUT / f"ts_oos_matrix_1w_lb{lb}_long.parquet") for lb in LOOKBACKS}
    ts = phase_two_stage(mats)
    ls = phase_control_ls(ts["chosen_lookback"])
    va = phase_value_added(mats[ts["chosen_lookback"]], ts["chosen_lookback"])
    write_report(ts, ls, va, mats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
