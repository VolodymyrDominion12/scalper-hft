"""iter14 — CS-momentum, 12-1 TSMOM, друга пара (two-stage).

Pre-registration: docs/reports/hypothesis_iter14.md (ЗАФІКСОВАНО до прогону).
Запуск: uv run python experiments/iter14_improvement_cycle.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter14"
OUT.mkdir(parents=True, exist_ok=True)

CORE_15 = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,"
    "AVAXUSDT,UNIUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,LTCUSDT,AAVEUSDT"
).split(",")

EXCLUDED_PAIRS = frozenset(
    {
        frozenset({"LINKUSDT", "BTCUSDT"}),
        frozenset({"XRPUSDT", "BTCUSDT"}),
        frozenset({"LINKUSDT", "ETHUSDT"}),
        frozenset({"BTCUSDT", "ETHUSDT"}),
        frozenset({"ETHUSDT", "SOLUSDT"}),
        frozenset({"BTCUSDT", "SOLUSDT"}),
    }
)

CS_LOOKBACKS = (10, 20, 60)
TOP_PCT = 0.20
SKIPMOM_LOOKBACK = 252
SKIPMOM_SKIP = 21
MIN_NAMES = 5
DAYS_1D = 2500
DAYS_1H = 1095
BARS_PER_YEAR_1D = 365.0
NW_LAGS = 20
SEL_START, SEL_END = "2019-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2026-12-31"
PAIRS_TRAIN, PAIRS_TEST = 1500, 500
MAKER_FEE = 0.0002


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


def _gate(port: pd.Series) -> dict:
    val = port.loc[VAL_START:VAL_END].dropna()
    s_val = sharpe(val.to_numpy())
    t_val = newey_west_t(val.to_numpy())
    ci_lo, ci_hi = bootstrap_ci(val.to_numpy())
    promote = bool(s_val > 0 and t_val >= 2.0 and ci_lo > 0)
    return {
        "val_sharpe": s_val,
        "val_tnw": t_val,
        "val_ci_lo": ci_lo,
        "val_ci_hi": ci_hi,
        "val_maxdd": max_dd(val.to_numpy()),
        "val_days": int(len(val)),
        "full_sharpe": sharpe(port.to_numpy()),
        "full_tnw": newey_west_t(port.to_numpy()),
        "promote": promote,
    }


def load_close_panel(symbols: list[str], interval: str, days: int, *, derive: bool) -> pd.DataFrame:
    from scalper_hft.data.access import ensure_klines

    cols: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for symbol in symbols:
        try:
            df = ensure_klines(symbol, interval, days, derive=derive)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{symbol}: {type(exc).__name__}")
            continue
        if df is None or df.empty or "close" not in df.columns:
            skipped.append(f"{symbol}: empty")
            continue
        cols[symbol] = df["close"].astype(float)
    if skipped:
        print(f"    skip panel: {skipped}", flush=True)
    panel = pd.DataFrame(cols).sort_index()
    return panel


def cs_weights(closes: pd.DataFrame, lookback: int, *, allow_short: bool) -> pd.DataFrame:
    mom = closes.pct_change(lookback)
    ranks = mom.rank(axis=1, pct=True, method="average")
    long_mask = (ranks >= (1.0 - TOP_PCT)) & mom.notna()
    n_long = long_mask.sum(axis=1)
    n_long = n_long.where(n_long >= 1, np.nan)
    w_long = long_mask.astype(float).div(n_long, axis=0).fillna(0.0)
    n_avail = mom.notna().sum(axis=1)
    w_long = w_long.where(n_avail >= MIN_NAMES, 0.0)
    if not allow_short:
        return w_long
    short_mask = (ranks <= TOP_PCT) & mom.notna()
    n_short = short_mask.sum(axis=1)
    n_short = n_short.where(n_short >= 1, np.nan)
    w_short = short_mask.astype(float).div(n_short, axis=0).fillna(0.0)
    w = 0.5 * w_long - 0.5 * w_short
    return w.where(n_avail >= MIN_NAMES, 0.0)


def skip_month_weights(closes: pd.DataFrame, lookback: int, skip: int, *, allow_short: bool) -> pd.DataFrame:
    mom = closes.shift(skip) / closes.shift(lookback) - 1.0
    sig = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    sig = sig.mask(mom > 0.0, 1.0)
    if allow_short:
        sig = sig.mask(mom < 0.0, -1.0)
    n_active = sig.abs().sum(axis=1)
    n_active = n_active.where(n_active >= 1, np.nan)
    n_avail = mom.notna().sum(axis=1)
    w = sig.div(n_active, axis=0).fillna(0.0)
    return w.where(n_avail >= MIN_NAMES, 0.0)


def portfolio_from_weights(closes: pd.DataFrame, weights: pd.DataFrame, fee: float = MAKER_FEE) -> pd.Series:
    ret = closes.pct_change()
    w_exec = weights.shift(1)
    aligned = w_exec.reindex(ret.index).fillna(0.0)
    gross = (aligned * ret).sum(axis=1)
    turnover = aligned.diff().abs().sum(axis=1)
    net = gross - turnover * fee
    return net.dropna()


def phase_cs(closes: pd.DataFrame) -> dict:
    print("=== iter14 H14-A: CS momentum two-stage ===", flush=True)
    sel_rows: list[dict] = []
    ports: dict[int, pd.Series] = {}
    for lb in CS_LOOKBACKS:
        w = cs_weights(closes, lb, allow_short=False)
        port = portfolio_from_weights(closes, w)
        ports[lb] = port
        sel = port.loc[SEL_START:SEL_END]
        row = {
            "lookback": lb,
            "sel_sharpe": sharpe(sel.to_numpy()),
            "sel_tnw": newey_west_t(sel.to_numpy()),
            "sel_days": int(len(sel)),
        }
        sel_rows.append(row)
        print(f"    lb={lb}: sel SR={row['sel_sharpe']:+.3f} t_NW={row['sel_tnw']:+.2f}", flush=True)
    sel_df = pd.DataFrame(sel_rows)
    chosen = int(sel_df.loc[sel_df.sel_sharpe.idxmax(), "lookback"])
    gate = _gate(ports[chosen])
    w_ls = cs_weights(closes, chosen, allow_short=True)
    port_ls = portfolio_from_weights(closes, w_ls)
    ls = _gate(port_ls)
    print(
        f"    chosen lb={chosen} val SR={gate['val_sharpe']:+.3f} t_NW={gate['val_tnw']:+.2f} "
        f"CI=[{gate['val_ci_lo']:+.2f},{gate['val_ci_hi']:+.2f}] promote={gate['promote']}",
        flush=True,
    )
    ports[chosen].to_frame("cs_long").to_parquet(OUT / "h14a_cs_long.parquet")
    port_ls.to_frame("cs_ls").to_parquet(OUT / "h14a_cs_ls.parquet")
    return {
        "hypothesis": "H14-A",
        "chosen_lookback": chosen,
        "sel_rows": sel_rows,
        **gate,
        "ls": ls,
        "port": ports[chosen],
        "port_ls": port_ls,
    }


def phase_skipmom(closes: pd.DataFrame) -> dict:
    print("=== iter14 H14-B: 12-1 skip-month TSMOM ===", flush=True)
    w = skip_month_weights(closes, SKIPMOM_LOOKBACK, SKIPMOM_SKIP, allow_short=False)
    port = portfolio_from_weights(closes, w)
    gate = _gate(port)
    w_ctrl = skip_month_weights(closes, SKIPMOM_LOOKBACK, 0, allow_short=False)
    ctrl = _gate(portfolio_from_weights(closes, w_ctrl))
    print(
        f"    12-1 val SR={gate['val_sharpe']:+.3f} t_NW={gate['val_tnw']:+.2f} "
        f"CI=[{gate['val_ci_lo']:+.2f},{gate['val_ci_hi']:+.2f}] promote={gate['promote']}",
        flush=True,
    )
    print(f"    control skip=0 val SR={ctrl['val_sharpe']:+.3f} t_NW={ctrl['val_tnw']:+.2f}", flush=True)
    port.to_frame("skipmom").to_parquet(OUT / "h14b_skipmom.parquet")
    return {"hypothesis": "H14-B", **gate, "control": ctrl, "port": port}


def phase_pairs(closes_1d: pd.DataFrame) -> dict:
    print("=== iter14 H14-C: IS coint scan → 1h validation ===", flush=True)
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs import run_pairs_backtest, run_pairs_walk_forward
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.strategies.pairs_arb import PairsArb
    from scalper_hft.validation.coint_scan import scan_pairs
    from scalper_hft.validation.pairs_gate import evaluate_pair_wf_gate

    sel = closes_1d.loc[SEL_START:SEL_END]
    closes_map = {c: sel[c].dropna() for c in sel.columns}
    rows = scan_pairs(closes_map)
    scan_df = pd.DataFrame(
        [
            {
                "leg1": r.leg1,
                "leg2": r.leg2,
                "adf_pvalue": r.adf_pvalue,
                "half_life": r.half_life,
                "tradable": r.tradable,
                "excluded": frozenset({r.leg1, r.leg2}) in EXCLUDED_PAIRS,
            }
            for r in rows
        ]
    )
    scan_df.to_csv(OUT / "coint_scan_selection.csv", index=False)
    candidates = [r for r in rows if r.tradable and frozenset({r.leg1, r.leg2}) not in EXCLUDED_PAIRS][:3]
    print(f"    tradable new pairs: {[(r.leg1, r.leg2, r.adf_pvalue, r.half_life) for r in candidates]}", flush=True)
    if not candidates:
        return {"hypothesis": "H14-C", "error": "no tradable new pairs on IS scan", "scan_n": int(len(rows))}

    settings = get_settings()
    results: list[dict] = []
    for cand in candidates:
        t0 = time.time()
        d1 = ensure_klines(cand.leg1, "1h", DAYS_1H, derive=True)
        d2 = ensure_klines(cand.leg2, "1h", DAYS_1H, derive=True)
        f1 = download_funding(cand.leg1, DAYS_1H)
        f2 = download_funding(cand.leg2, DAYS_1H)
        strat = PairsArb(entry_z=2.0, exit_z=0.3, lookback=120, regime_scale=True, regime_scale_factor=0.25)
        cost = CostModel.from_settings(settings, df=d1)
        wf = run_pairs_walk_forward(
            d1, d2, strat, f1, f2, train_bars=PAIRS_TRAIN, test_bars=PAIRS_TEST, position_pct=0.1, maker_execution=True
        )
        cut = pd.Timestamp(VAL_START)
        d1v, d2v = d1[d1.index >= cut], d2[d2.index >= cut]
        res_val = run_pairs_backtest(d1v, d2v, strat, f1, f2, position_pct=0.1, cost=cost, maker_execution=True)
        m = res_val.metrics
        wf_label, wf_reasons = evaluate_pair_wf_gate(wf["positive_windows"], wf["n_windows"])
        promote = bool(
            wf_label == "PASS" and wf["n_windows"] >= 3 and int(m.n_trades) >= 20 and float(m.total_return) > 0
        )
        row = {
            "pair": f"{cand.leg1}/{cand.leg2}",
            "adf_p": cand.adf_pvalue,
            "half_life_1d": cand.half_life,
            "wf_pos": wf["positive_windows"],
            "n_windows": wf["n_windows"],
            "wf_avg_oos_srh": wf["avg_oos_sharpe"],
            "val_ret": float(m.total_return),
            "val_pf": float(m.profit_factor),
            "val_trades": int(m.n_trades),
            "val_maxdd": float(m.max_drawdown),
            "wf_gate": wf_label,
            "wf_reasons": "; ".join(wf_reasons),
            "promote": promote,
            "seconds": round(time.time() - t0, 1),
        }
        results.append(row)
        print(
            f"    {row['pair']}: WF pos={row['wf_pos']:.0%} n={row['n_windows']} ret={row['val_ret']:+.1%} trades={row['val_trades']} promote={promote}",
            flush=True,
        )
    pd.DataFrame(results).to_csv(OUT / "h14c_pairs.csv", index=False)
    return {"hypothesis": "H14-C", "pairs": results, "scan_n": int(len(rows))}


def phase_value_added(cs: dict, skip: dict) -> dict:
    print("=== iter14 H14-D: value-added vs ts_momentum 1d ===", flush=True)
    path_1d = ROOT / "results" / "iter12" / "ts_oos_matrix_1d.parquet"
    if not path_1d.exists():
        return {"hypothesis": "H14-D", "error": "немає матриці iter12 1d"}
    ts = pd.read_parquet(path_1d).mean(axis=1).dropna()
    out: dict = {"hypothesis": "H14-D"}
    for name, blob in (("cs", cs), ("skipmom", skip)):
        if blob.get("val_sharpe", 0) <= 0:
            out[name] = {"skipped": "validation Sharpe <= 0"}
            continue
        port: pd.Series = blob["port"]
        common = ts.index.intersection(port.index)
        if len(common) < 100:
            out[name] = {"error": f"overlap {len(common)}"}
            continue
        a, b = port.loc[common], ts.loc[common]
        combined = 0.5 * a + 0.5 * b
        rec = {
            "overlap_days": int(len(common)),
            "alpha_sharpe": sharpe(a.to_numpy()),
            "ts_sharpe": sharpe(b.to_numpy()),
            "combined_sharpe": sharpe(combined.to_numpy()),
            "alpha_maxdd": max_dd(a.to_numpy()),
            "ts_maxdd": max_dd(b.to_numpy()),
            "combined_maxdd": max_dd(combined.to_numpy()),
            "corr": float(a.corr(b)),
        }
        rec["adds_value"] = bool(
            rec["combined_sharpe"] > max(rec["alpha_sharpe"], rec["ts_sharpe"])
            and rec["combined_maxdd"] > max(rec["alpha_maxdd"], rec["ts_maxdd"])  # less negative DD
        )
        out[name] = rec
        print(
            f"    {name}: corr={rec['corr']:+.2f} combined SR={rec['combined_sharpe']:+.2f} adds={rec['adds_value']}",
            flush=True,
        )
    return out


def _variant_row(
    *, variant: str, strategy: str, interval: str, gate: dict, notes: str, mean_oos: float | None = None
) -> dict:
    return {
        "variant": variant,
        "strategy": strategy,
        "interval": interval,
        "mean_oos_wf": mean_oos if mean_oos is not None else gate.get("val_sharpe"),
        "port_sharpe": gate.get("val_sharpe"),
        "t_newey_west": gate.get("val_tnw"),
        "symbols_pos_frac": None,
        "n_trades_oos": None,
        "notes": notes,
    }


def write_artifacts(cs: dict, skip: dict, pairs: dict, va: dict) -> None:
    print("=== iter14: звіт + leaderboard artifacts ===", flush=True)
    from scalper_hft.research.leaderboard import leaderboard_dataframe, render_leaderboard_markdown
    from scalper_hft.validation.oos_registry import append_usage
    from scalper_hft.validation.oos_registry import default_path as oos_path
    from scalper_hft.validation.trial_ledger import default_path as ledger_path
    from scalper_hft.validation.trial_ledger import record_trial

    variants: list[dict] = []
    chosen = cs["chosen_lookback"]
    cs_notes = (
        f"H14-A two-stage val; lookback={chosen}; short=False; "
        f"{'PROMOTE monitoring' if cs['promote'] else 'FAIL gate'}; n=15 1d CS"
    )
    variants.append(
        _variant_row(variant=f"cs_long_lb{chosen}", strategy="cross_momentum", interval="1d", gate=cs, notes=cs_notes)
    )
    ls_notes = f"H14-A control LS; lookback={chosen}; short=True; n=15 1d CS"
    variants.append(
        _variant_row(
            variant=f"cs_ls_lb{chosen}", strategy="cross_momentum", interval="1d", gate=cs["ls"], notes=ls_notes
        )
    )
    skip_notes = (
        f"H14-B 12-1 skip-month; lookback=252 skip=21; short=False; "
        f"{'PROMOTE monitoring' if skip['promote'] else 'FAIL gate'}"
    )
    variants.append(
        _variant_row(variant="skipmom_12_1", strategy="ts_momentum", interval="1d", gate=skip, notes=skip_notes)
    )
    ctrl_notes = "H14-B control skip=0 lookback=252; short=False; FAIL gate"
    variants.append(
        _variant_row(
            variant="skipmom_12_0", strategy="ts_momentum", interval="1d", gate=skip["control"], notes=ctrl_notes
        )
    )
    for rec in pairs.get("pairs", []):
        verdict = "CANDIDATE" if rec["promote"] else "REJECTED"
        variants.append(
            {
                "variant": rec["pair"].replace("USDT", "").replace("/", "_"),
                "strategy": "pairs_arb",
                "interval": "1h",
                "mean_oos_wf": rec["wf_avg_oos_srh"],
                "port_sharpe": None,
                "t_newey_west": None,
                "symbols_pos_frac": rec["wf_pos"],
                "n_trades_oos": rec["val_trades"],
                "notes": (
                    f"H14-C {verdict}; WF pos={rec['wf_pos']:.0%} n={rec['n_windows']}; "
                    f"val ret={rec['val_ret']:+.1%} trades={rec['val_trades']}"
                ),
            }
        )
    pd.DataFrame(variants).to_csv(OUT / "variants.csv", index=False)

    record_trial(
        ledger_path(),
        strategy="cross_momentum",
        symbol="PORTFOLIO_CORE15",
        purpose="iter14:cs/1d",
        n_trials=len(CS_LOOKBACKS),
    )
    record_trial(
        ledger_path(), strategy="ts_momentum", symbol="PORTFOLIO_CORE15", purpose="iter14:skipmom-12-1/1d", n_trials=2
    )
    record_trial(
        ledger_path(),
        strategy="pairs_arb",
        symbol="PORTFOLIO_NEWPAIRS",
        purpose="iter14:coint-scan/1h",
        n_trials=max(len(pairs.get("pairs", [])), 1),
    )
    from datetime import date

    append_usage(oos_path(), "cross_momentum", "PORTFOLIO_CORE15", date(2019, 11, 8), date(2026, 9, 12), "iter14:cs/1d")
    append_usage(
        oos_path(),
        "ts_momentum",
        "PORTFOLIO_SKIPMOM_12_1",
        date(2019, 11, 8),
        date(2026, 9, 12),
        "iter14:skipmom-12-1/1d",
    )
    append_usage(
        oos_path(), "pairs_arb", "NEWPAIRS_CORE15", date(2023, 9, 12), date(2026, 9, 12), "iter14:coint-scan/1h"
    )

    md = _render_report(cs, skip, pairs, va)
    report = ROOT / "docs" / "reports" / "iter14_improvement_cycle.md"
    report.write_text(md, encoding="utf-8")
    payload = {
        "cs": {k: v for k, v in cs.items() if k not in {"port", "port_ls"}},
        "skip": {k: v for k, v in skip.items() if k != "port"},
        "pairs": pairs,
        "va": va,
    }
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")

    lb_md = render_leaderboard_markdown(ROOT / "results")
    (ROOT / "docs" / "reports" / "LEADERBOARD.md").write_text(lb_md, encoding="utf-8")
    leaderboard_dataframe(ROOT / "results").to_csv(OUT / "leaderboard.csv", index=False)
    print(f"Звіт: {report}", flush=True)


def _fmt_ci(lo: float, hi: float) -> str:
    if lo != lo or hi != hi:  # NaN
        return "n/a"
    return f"[{lo:+.2f}, {hi:+.2f}]"


def _render_report(cs: dict, skip: dict, pairs: dict, va: dict) -> str:
    lines: list[str] = []
    lines.append("# iter14 — CS-momentum, 12-1 TSMOM, друга пара (two-stage)\n")
    lines.append("Дата: 2026-09-12 · Pre-registration: [hypothesis_iter14.md](hypothesis_iter14.md)")
    lines.append("Дані: LIVE `binanceusdm`, нативні 1d CORE_15 + 1h validation пар.\n")
    lines.append("## TL;DR\n")
    lines.append("| Гіпотеза | Результат | Деталі |")
    lines.append("|---|---|---|")
    a_v = "PASS → monitoring" if cs["promote"] else "FAIL"
    lines.append(
        f"| H14-A CS momentum long-only | {a_v} | lookback={cs['chosen_lookback']}d; "
        f"validation SR {cs['val_sharpe']:+.3f}, t_NW {cs['val_tnw']:+.2f}, CI {_fmt_ci(cs['val_ci_lo'], cs['val_ci_hi'])} |"
    )
    b_v = "PASS → monitoring" if skip["promote"] else "FAIL"
    lines.append(
        f"| H14-B 12-1 skip-month TSMOM | {b_v} | validation SR {skip['val_sharpe']:+.3f}, "
        f"t_NW {skip['val_tnw']:+.2f}, CI {_fmt_ci(skip['val_ci_lo'], skip['val_ci_hi'])} |"
    )
    if "error" in pairs:
        lines.append(f"| H14-C друга пара | FAIL | {pairs['error']} |")
    else:
        any_p = any(r["promote"] for r in pairs.get("pairs", []))
        best = max(pairs.get("pairs", []), key=lambda r: r["wf_pos"], default=None)
        det = f"{best['pair']} WF pos={best['wf_pos']:.0%} ret={best['val_ret']:+.1%}" if best else "немає пар"
        lines.append(f"| H14-C друга пара | {'PASS → candidate' if any_p else 'FAIL'} | {det} |")
    lines.append("")
    lines.append("## H14-A: selection half (2019–2022)\n")
    lines.append("| lookback | selection Sharpe | t_NW | днів |")
    lines.append("|---:|---:|---:|---:|")
    for r in cs["sel_rows"]:
        lines.append(f"| {r['lookback']} | {r['sel_sharpe']:+.3f} | {r['sel_tnw']:+.2f} | {r['sel_days']} |")
    lines.append(f"\nОбрано lookback = **{cs['chosen_lookback']}** днів.\n")
    lines.append("## H14-A: validation half (2023–2026)\n")
    lines.append("| Метрика | long-only | long-short контроль |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Sharpe | {cs['val_sharpe']:+.3f} | {cs['ls']['val_sharpe']:+.3f} |")
    lines.append(f"| t_Newey–West(20) | {cs['val_tnw']:+.2f} | {cs['ls']['val_tnw']:+.2f} |")
    lines.append(
        f"| bootstrap 95% CI | {_fmt_ci(cs['val_ci_lo'], cs['val_ci_hi'])} | {_fmt_ci(cs['ls']['val_ci_lo'], cs['ls']['val_ci_hi'])} |"
    )
    lines.append(f"| maxDD | {cs['val_maxdd']:+.2%} | {cs['ls']['val_maxdd']:+.2%} |")
    lines.append(
        f"\nРішення: validation Sharpe>0 AND t_NW≥2.0 AND CI без 0 → "
        f"**{'PROMOTE → monitoring' if cs['promote'] else 'НЕ промотувати'}**.\n"
    )
    lines.append("## H14-B: 12-1 skip-month (validation 2023–2026)\n")
    lines.append("| Варіант | Sharpe | t_NW | CI | maxDD |")
    lines.append("|---|---:|---:|:---|---:|")
    lines.append(
        f"| 12-1 (skip=21) | {skip['val_sharpe']:+.3f} | {skip['val_tnw']:+.2f} | "
        f"{_fmt_ci(skip['val_ci_lo'], skip['val_ci_hi'])} | {skip['val_maxdd']:+.2%} |"
    )
    lines.append(
        f"| контроль skip=0 | {skip['control']['val_sharpe']:+.3f} | {skip['control']['val_tnw']:+.2f} | "
        f"{_fmt_ci(skip['control']['val_ci_lo'], skip['control']['val_ci_hi'])} | {skip['control']['val_maxdd']:+.2%} |"
    )
    lines.append(f"\nРішення: **{'PROMOTE → monitoring' if skip['promote'] else 'НЕ промотувати'}**.\n")
    lines.append("## H14-C: нові пари\n")
    if "error" in pairs:
        lines.append(f"{pairs['error']}\n")
    else:
        lines.append("| Пара | ADF p (IS) | HL 1d | WF pos | вікон | val ret | угод | maxDD | гейт |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---|")
        for r in pairs.get("pairs", []):
            flag = "PASS" if r["promote"] else "FAIL"
            lines.append(
                f"| {r['pair']} | {r['adf_p']:.3f} | {r['half_life_1d']:.1f} | {r['wf_pos']:.0%} | "
                f"{r['n_windows']} | {r['val_ret']:+.1%} | {r['val_trades']} | {r['val_maxdd']:+.1%} | {flag} |"
            )
        lines.append("\nPASS → candidate (не `VALIDATED_PAIRS`; CSCV перед будь-якою зміною списку).\n")
    if "error" not in va:
        lines.append("## H14-D: value-added vs ts_momentum 1d\n")
        for name, rec in va.items():
            if name == "hypothesis" or not isinstance(rec, dict) or "combined_sharpe" not in rec:
                continue
            lines.append(
                f"- **{name}**: corr={rec['corr']:+.2f}; standalone SR {rec['alpha_sharpe']:+.2f} vs ts {rec['ts_sharpe']:+.2f}; "
                f"combined {rec['combined_sharpe']:+.2f}; adds={rec['adds_value']}"
            )
        lines.append("")
    lines.append("## Вердикт для paper\n")
    promoted: list[str] = []
    if cs["promote"]:
        promoted.append(f"cross-sectional momentum 1d long-only lookback={cs['chosen_lookback']} (monitoring)")
    if skip["promote"]:
        promoted.append("12-1 skip-month TSMOM 1d long-only (monitoring)")
    for r in pairs.get("pairs", []):
        if r["promote"]:
            promoted.append(f"pairs_arb {r['pair']} 1h (candidate, не validated)")
    if promoted:
        lines.append("Нові комірки, що пройшли pre-registered гейт:")
        for p in promoted:
            lines.append(f"- {p}")
        lines.append("Ядро paper без змін: **pairs_arb LINK/BTC 1h maker** (validated) + ts_momentum 1d/4h monitoring.")
    else:
        lines.append(
            "Жодна нова гіпотеза не пройшла гейт. Paper стартує на наявному наборі: "
            "**pairs_arb LINK/BTC 1h maker** (validated) + **ts_momentum 1d/4h long-only CORE_15** (monitoring). "
            "Новий research-цикл — лише з новою pre-registration."
        )
    lines.append("\n⚠ Вікна після цього прогону спалені: див. `docs/reports/oos_usage.md`.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "panel", "pairs", "report"], default="all")
    args = ap.parse_args()
    print("=== iter14: load 1d panel ===", flush=True)
    closes = load_close_panel(CORE_15, "1d", DAYS_1D, derive=False)
    closes.to_parquet(OUT / "closes_1d.parquet")
    print(f"    panel {closes.shape} {closes.index[0]} → {closes.index[-1]}", flush=True)
    cs = phase_cs(closes)
    skip = phase_skipmom(closes)
    if args.phase in ("all", "pairs"):
        pairs = phase_pairs(closes)
    else:
        pairs = {"hypothesis": "H14-C", "error": "skipped"}
    va = phase_value_added(cs, skip)
    write_artifacts(cs, skip, pairs, va)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
