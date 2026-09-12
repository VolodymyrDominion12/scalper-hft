"""iter12 — improvement loop: two-stage vol-target overlay + value-added.

Pre-registration: docs/reports/hypothesis_iter12.md (ЗАФІКСОВАНО до прогону).
Запуск: uv run python experiments/iter12_improvement_cycle.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter12"
OUT.mkdir(parents=True, exist_ok=True)

CORE_15 = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,"
    "AVAXUSDT,UNIUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,LTCUSDT,AAVEUSDT"
).split(",")

TRAIN_1D, TEST_1D = 250, 125
BARS_PER_YEAR_1D = 365.0
DAYS = 2500
NW_LAGS = 20
VOL_WINDOWS = (20, 40, 60)
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


def sharpe(returns: np.ndarray, bpy: float = BARS_PER_YEAR_1D) -> float:
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
    returns: np.ndarray, bpy: float = BARS_PER_YEAR_1D, n_boot: int = 2000, seed: int = 42
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


def vol_target_overlay(port: pd.Series, vol_window: int) -> pd.Series:
    realized = port.rolling(vol_window, min_periods=10).std(ddof=0).shift(1)
    med_vol = float(realized.median())
    if med_vol <= 0:
        return port
    w = (med_vol / realized).clip(0.0, 2.0).fillna(0.0)
    return (port * w).dropna()


def phase_ts_returns() -> pd.DataFrame:
    print("=== iter12 фаза 1: ts_momentum 1d CORE_15 per-symbol OOS ===", flush=True)
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.walk_forward import run_walk_forward

    t0 = time.time()
    strategy = get_strategy("ts_momentum", lookback=20, signal_smooth=1, allow_short=False)
    cost = CostModel.from_settings(get_settings())
    per_symbol: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for symbol in CORE_15:
        try:
            df = ensure_klines(symbol, "1d", DAYS, derive=False)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{symbol}: {type(exc).__name__}")
            continue
        if df is None or len(df) < TRAIN_1D + TEST_1D + 10:
            skipped.append(f"{symbol}: bars={0 if df is None else len(df)}")
            continue
        wf = run_walk_forward(
            df,
            strategy,
            train_bars=TRAIN_1D,
            test_bars=TEST_1D,
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
            skipped.append(f"{symbol}: empty OOS")
            continue
        per_symbol[symbol] = oos
    if skipped:
        print(f"    skip: {skipped}", flush=True)
    mat = pd.DataFrame(per_symbol)
    mat.to_parquet(OUT / "ts_oos_matrix_1d.parquet")
    print(f"    матриця {mat.shape} збережена ({time.time() - t0:.0f}s)", flush=True)
    return mat


def phase_two_stage_voltarget(mat: pd.DataFrame) -> dict:
    print("=== iter12 фаза 2 (H12-A): two-stage vol-target ===", flush=True)
    port = mat.mean(axis=1).dropna()
    sel_rows = [
        {"vol_window": w, "sel_sharpe": sharpe(vol_target_overlay(port, w).loc[SEL_START:SEL_END].to_numpy())}
        for w in VOL_WINDOWS
    ]
    sel_df = pd.DataFrame(sel_rows)
    chosen = int(sel_df.loc[sel_df.sel_sharpe.idxmax(), "vol_window"])
    print(f"    selection sharpes: {sel_rows} | обрано σ-вікно={chosen}", flush=True)
    vt_full = vol_target_overlay(port, chosen)
    val_vt = vt_full.loc[VAL_START:VAL_END]
    val_plain = port.loc[VAL_START:VAL_END]
    s_plain, s_vt = sharpe(val_plain.to_numpy()), sharpe(val_vt.to_numpy())
    t_plain, t_vt = newey_west_t(val_plain.to_numpy()), newey_west_t(val_vt.to_numpy())
    ci_lo, ci_hi = bootstrap_ci(val_vt.to_numpy())
    promote = bool(s_vt > s_plain and t_vt >= 2.0)
    print(
        f"    validation: plain SR={s_plain:+.3f} t_NW={t_plain:+.2f} | "
        f"vt SR={s_vt:+.3f} t_NW={t_vt:+.2f} CI=[{ci_lo:+.2f},{ci_hi:+.2f}] | promote={promote}",
        flush=True,
    )
    return {
        "hypothesis": "H12-A",
        "chosen_vol_window": chosen,
        "sel_sharpes": sel_rows,
        "val_sharpe_plain": s_plain,
        "val_sharpe_vt": s_vt,
        "val_tnw_plain": t_plain,
        "val_tnw_vt": t_vt,
        "val_ci_lo": ci_lo,
        "val_ci_hi": ci_hi,
        "val_maxdd_plain": max_dd(val_plain.to_numpy()),
        "val_maxdd_vt": max_dd(val_vt.to_numpy()),
        "promote_to_monitoring": promote,
        "decision_rule": "vt Sharpe > plain AND t_NW(vt) >= 2.0 on validation half",
    }


def phase_attribution(mat: pd.DataFrame) -> pd.DataFrame:
    print("=== iter12 фаза 3 (H12-C): per-symbol attribution ===", flush=True)
    port = mat.mean(axis=1).dropna()
    port_std = float(port.std(ddof=1)) or 1.0
    rows = []
    for sym in mat.columns:
        s = mat[sym].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "symbol": sym,
                "mean_oos": float(s.mean()),
                "sharpe": sharpe(s.to_numpy()),
                "contribution": float(s.mean()) / port_std,
                "corr_to_port": float(s.corr(port)),
                "n_bars": int(len(s)),
            }
        )
    df = pd.DataFrame(rows).sort_values("contribution", ascending=False)
    df.to_csv(OUT / "attribution_1d.csv", index=False)
    print(f"    топ-3: {df.head(3)[['symbol', 'contribution', 'sharpe']].to_dict('records')}", flush=True)
    print(f"    worst-3: {df.tail(3)[['symbol', 'contribution', 'sharpe']].to_dict('records')}", flush=True)
    return df


def phase_value_added(mat: pd.DataFrame) -> dict:
    print("=== iter12 фаза 4 (H12-B): value-added pairs ⊕ ts ===", flush=True)
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs import run_pairs_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.strategies.pairs_arb import PairsArb

    t0 = time.time()
    leg1, leg2 = "LINKUSDT", "BTCUSDT"
    # pairs_arb валідовано на 3y 1h (не 6.8y); ts-матриця дає 2023-2026 для overlap.
    pairs_days = 1095
    df1 = ensure_klines(leg1, "1h", pairs_days, derive=True)
    df2 = ensure_klines(leg2, "1h", pairs_days, derive=True)
    f1 = download_funding(leg1, pairs_days)
    f2 = download_funding(leg2, pairs_days)
    strat = PairsArb(entry_z=2.0, exit_z=0.3, lookback=120, regime_scale=True, regime_scale_factor=0.25)
    cost = CostModel.from_settings(get_settings(), df=df1)
    res = run_pairs_backtest(df1, df2, strat, f1, f2, position_pct=0.1, cost=cost, maker_execution=True)
    pairs_ret = res.equity.pct_change().fillna(0.0)
    pairs_daily = pairs_ret.resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()
    ts_port = mat.mean(axis=1).dropna()
    common = pairs_daily.index.intersection(ts_port.index)
    if len(common) < 100:
        return {"hypothesis": "H12-B", "error": f"overlap too short: {len(common)}"}
    p, t = pairs_daily.loc[common], ts_port.loc[common]
    # 50/50 капітальна алокaція (raw equal-weight), не vol-нормалізація (вона множить плеча).
    combined = 0.5 * p + 0.5 * t
    bpy = BARS_PER_YEAR_1D
    out = {
        "hypothesis": "H12-B",
        "overlap_days": int(len(common)),
        "pairs_sharpe": sharpe(p.to_numpy(), bpy),
        "ts_sharpe": sharpe(t.to_numpy(), bpy),
        "combined_sharpe": sharpe(combined.to_numpy(), bpy),
        "pairs_maxdd": max_dd(p.to_numpy()),
        "ts_maxdd": max_dd(t.to_numpy()),
        "combined_maxdd": max_dd(combined.to_numpy()),
        "pairs_ts_corr": float(p.corr(t)),
    }
    out["diversification_adds_value"] = bool(
        out["combined_sharpe"] > max(out["pairs_sharpe"], out["ts_sharpe"])
        and out["combined_maxdd"] > max(out["pairs_maxdd"], out["ts_maxdd"])
    )
    print(
        f"    overlap={out['overlap_days']}d corr={out['pairs_ts_corr']:+.2f} | "
        f"pairs SR={out['pairs_sharpe']:+.2f} DD={out['pairs_maxdd']:+.2%} | "
        f"ts SR={out['ts_sharpe']:+.2f} DD={out['ts_maxdd']:+.2%} | "
        f"combined SR={out['combined_sharpe']:+.2f} DD={out['combined_maxdd']:+.2%} | "
        f"adds={out['diversification_adds_value']} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return out


def write_report(mat: pd.DataFrame, vt: dict, attr: pd.DataFrame, va: dict) -> str:
    print("=== iter12 фаза 5: звіт ===", flush=True)
    port = mat.mean(axis=1).dropna()
    overall_plain = sharpe(port.to_numpy())
    overall_t = newey_west_t(port.to_numpy())
    md: list[str] = []
    md.append("# iter12 — improvement loop: two-stage vol-target + value-added\n")
    md.append("Дата: 2026-09-12 · Pre-registration: [hypothesis_iter12.md](hypothesis_iter12.md)")
    md.append("Дані: LIVE `binanceusdm`, перевикористання спаленого OOS iter11 (CORE_15 1d).\n")
    md.append("## TL;DR\n")
    md.append("| Гіпотеза | Результат | Деталі |")
    md.append("|---|---|---|")
    vt_verdict = "PROMOTE → monitoring" if vt["promote_to_monitoring"] else "НЕ промотувати (post-hoc лишається)"
    md.append(
        f"| H12-A two-stage vol-target | {'PASS' if vt['promote_to_monitoring'] else 'FAIL'} | {vt_verdict}; обране σ={vt['chosen_vol_window']} |"
    )
    va_verdict = "диверсифікація додає вартість" if va.get("diversification_adds_value") else "без доданої вартості"
    md.append(
        f"| H12-B value-added pairs⊕ts | {'PASS' if va.get('diversification_adds_value') else 'FAIL'} | {va_verdict}; corr={va.get('pairs_ts_corr', 0):+.2f} |"
    )
    md.append(
        f"| H12-C attribution | діагностика | топ: {attr.head(3)['symbol'].tolist()} | worst: {attr.tail(3)['symbol'].tolist()} |\n"
    )
    md.append(
        f"Портфель ts_momentum 1d CORE_15 (повний OOS): Sharpe **{overall_plain:+.2f}**, t_NW **{overall_t:+.2f}**.\n"
    )
    md.append("## H12-A: two-stage vol-target overlay\n")
    md.append(f"Сітка σ-вікон (заздалегідь): {list(VOL_WINDOWS)}.")
    md.append(
        f"Selection half {SEL_START}→{SEL_END} обирає вікно з макс. Sharpe; validation half {VAL_START}→{VAL_END} — невидиме під час selection.\n"
    )
    md.append("| σ-вікно | selection Sharpe |")
    md.append("|---:|---:|")
    for r in vt["sel_sharpes"]:
        md.append(f"| {r['vol_window']} | {r['sel_sharpe']:+.3f} |")
    md.append(f"\nОбране σ-вікно: **{vt['chosen_vol_window']}**\n")
    md.append("| Метрика | plain | vol-target |")
    md.append("|---|---:|---:|")
    md.append(f"| validation Sharpe | {vt['val_sharpe_plain']:+.3f} | {vt['val_sharpe_vt']:+.3f} |")
    md.append(f"| validation t_NW | {vt['val_tnw_plain']:+.2f} | {vt['val_tnw_vt']:+.2f} |")
    md.append(f"| validation CI | — | [{vt['val_ci_lo']:+.2f}, {vt['val_ci_hi']:+.2f}] |")
    md.append(f"| validation maxDD | {vt['val_maxdd_plain']:+.2%} | {vt['val_maxdd_vt']:+.2%} |")
    md.append(
        f"\nРішення (заздалегідь): vt Sharpe > plain AND t_NW(vt) ≥ 2.0 на validation → **{'PROMOTE' if vt['promote_to_monitoring'] else 'НЕ промотувати'}**.\n"
    )
    md.append("## H12-B: value-added pairs_arb ⊕ ts_momentum\n")
    if "error" in va:
        md.append(f"Помилка: {va['error']}")
    else:
        md.append(f"Спільний span: {va['overlap_days']} днів; кореляція pairs↔ts = **{va['pairs_ts_corr']:+.2f}**.\n")
        md.append("| Портфель | Sharpe | maxDD |")
        md.append("|---|---:|---:|")
        md.append(f"| pairs_arb LINK/BTC 1h | {va['pairs_sharpe']:+.2f} | {va['pairs_maxdd']:+.2%} |")
        md.append(f"| ts_momentum 1d CORE_15 | {va['ts_sharpe']:+.2f} | {va['ts_maxdd']:+.2%} |")
        md.append(f"| combined 50/50 risk-equal | {va['combined_sharpe']:+.2f} | {va['combined_maxdd']:+.2%} |")
        md.append(
            f"\nВердикт: **{'диверсифікація додає вартість' if va['diversification_adds_value'] else 'без доданої вартості'}** (portfolio-construction evidence, не paper-gate).\n"
        )
    md.append("## H12-C: per-symbol attribution\n")
    md.append("| symbol | contribution | Sharpe | corr→port |")
    md.append("|---|---:|---:|---:|")
    for _, r in attr.iterrows():
        md.append(f"| {r['symbol']} | {r['contribution']:+.4f} | {r['sharpe']:+.2f} | {r['corr_to_port']:+.2f} |")
    md.append("\n## Висновки для paper\n")
    md.append("- `pairs_arb LINK/BTC 1h` — validated, без змін.")
    md.append("- `ts_momentum 1d/4h long-only CORE_15` — monitoring (iter11), без змін.")
    if vt["promote_to_monitoring"]:
        md.append(
            f"- vol-target overlay (σ={vt['chosen_vol_window']}) → **monitoring** (two-stage PASS): додає на validation половині."
        )
    else:
        md.append("- vol-target overlay → лишається пост-хок (two-stage FAIL на validation).")
    md.append("\n⚠ OOS перевикористано з iter11 (two-stage protocol); нового снупінгу немає.")
    out_md = ROOT / "docs" / "reports" / "iter12_improvement_cycle.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    (OUT / "results.json").write_text(
        json.dumps({"vt": vt, "va": va}, ensure_ascii=False, default=str, indent=2), encoding="utf-8"
    )
    print(f"Звіт: {out_md}", flush=True)
    return str(out_md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "ts", "voltarget", "attr", "value", "report"], default="all")
    args = ap.parse_args()
    mat = phase_ts_returns() if args.phase in ("all", "ts") else pd.read_parquet(OUT / "ts_oos_matrix_1d.parquet")
    vt = phase_two_stage_voltarget(mat) if args.phase in ("all", "voltarget") else {}
    attr = phase_attribution(mat) if args.phase in ("all", "attr") else pd.DataFrame()
    va = phase_value_added(mat) if args.phase in ("all", "value") else {}
    if args.phase in ("all", "report"):
        write_report(mat, vt, attr, va)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
