"""Аналіз повної матриці sweep та аудит топ-1 стратегії на кожну комірку (символ×ТФ).

Етапи:
    stage=winners  — з results/sweep.db (mode=walkforward, days=180) обрати
                     топ-1 за avg_oos_sharpe на кожну (symbol, interval);
                     зберегти results/full_matrix/winners_per_cell.csv
    stage=audit    — для кожної комірки з winners: WF (ті самі train/test, що у
                     sweep), sensitivity головного параметра (якщо є), Deflated
                     Sharpe → results/full_matrix/audits.csv та .md
    stage=report   — звести sweep + audits у фінальний docs/reports звіт.

Вердикт комірки (за критеріями overfitting-audit skill):
    PASS  якщо avg_oos_sharpe > 0.3, oos_positive_frac >= 0.5, DSR > 0.95,
          smoothness > 0.30, n_trades >= 100 (1m) / >= 40 (інші ТФ)
    інакше FAIL з поясненням.

Запуск:  EXCHANGE=binance .venv/bin/python scripts/matrix_audit.py winners|audit|report
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("matrix_audit")

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DB = ROOT / "results" / "sweep.db"
OUT_DIR = ROOT / "results" / "full_matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = [
    "mean_reversion", "market_maker", "funding_carry", "basis_reversion",
    "hmm_reversion", "cross_momentum", "supertrend", "stoch_rsi", "smc_fvg",
]
SYMBOLS = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,ADAUSDT,DOGEUSDT,"
    "AVAXUSDT,NEARUSDT,DOTUSDT,ATOMUSDT,UNIUSDT,LTCUSDT,AAVEUSDT"
).split(",")
INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h"]
DAYS = 180

# ті самі train/test, що у WF sweep (див. scripts/matrix_wf_sweeps.sh)
WF_TRAIN_TEST = {"1m": (4000, 2000), "5m": (2000, 1000), "15m": (1000, 500),
                 "30m": (500, 250), "1h": (500, 200), "4h": (200, 100)}
MIN_TRADES = {"1m": 100, "5m": 60, "15m": 40, "30m": 40, "1h": 30, "4h": 20}


def load_sweep(mode: str) -> pd.DataFrame:
    import sqlite3
    con = sqlite3.connect(SWEEP_DB)
    df = pd.read_sql_query(
        "SELECT * FROM sweep_results WHERE days=? AND mode=?", con, params=(DAYS, mode)
    )
    con.close()
    return df


def stage_winners() -> pd.DataFrame:
    wf = load_sweep("walkforward")
    ok = wf[wf["status"] == "ok"].copy()
    if ok.empty:
        print("Немає ok-клітинок у walkforward sweep — спершу запустіть WF матрицю")
        sys.exit(1)
    ok["oos_ok"] = ok["avg_oos_sharpe"].fillna(-99) > 0
    rows = []
    for (sym, iv), grp in ok.groupby(["symbol", "interval"]):
        best = grp.sort_values("avg_oos_sharpe", ascending=False).iloc[0]
        top3 = grp.sort_values("avg_oos_sharpe", ascending=False).head(3)
        rows.append({
            "symbol": sym, "interval": iv,
            "winner_strategy": best["strategy"],
            "winner_avg_oos_sharpe": best["avg_oos_sharpe"],
            "winner_oos_pos_frac": best["oos_positive_frac"],
            "winner_avg_is_sharpe": best["avg_is_sharpe"],
            "winner_n_trades_oos": best["n_trades"],
            "n_candidates_ok": len(grp),
            "n_candidates_oos_pos": int(grp["oos_ok"].sum()),
            "top3": json.dumps(
                [{"strategy": r["strategy"], "avg_oos_sharpe": float(r["avg_oos_sharpe"]),
                  "oos_pos_frac": float(r["oos_positive_frac"])} for _, r in top3.iterrows()]
            ),
        })
    out = pd.DataFrame(rows).sort_values(["interval", "symbol"])
    out.to_csv(OUT_DIR / "winners_per_cell.csv", index=False)
    print(out.to_markdown(index=False))
    print(f"\nЗбережено winners_per_cell.csv ({len(out)} комірок)")
    return out


def _cell_audit(task: dict) -> dict:
    """Один повний аудит комірки (у воркер-процесі)."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
    from scalper_hft.validation.sensitivity import parameter_sensitivity
    from scalper_hft.validation.walk_forward import run_walk_forward

    symbol, interval, name = task["symbol"], task["interval"], task["strategy"]
    res = {"symbol": symbol, "interval": interval, "strategy": name, "status": "error", "error": ""}
    try:
        settings = get_settings()
        cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee,
                         slippage_frac=settings.slippage_frac)
        strategy = get_strategy(name)
        df = ensure_klines(symbol, interval, DAYS)
        if df is None or df.empty:
            res["error"] = "немає даних"
            return res
        funding = download_funding(symbol, DAYS) if strategy.needs_funding else None
        train, test = WF_TRAIN_TEST[interval]

        wf = run_walk_forward(df, strategy, train, test, cost=cost, funding=funding,
                              position_pct=settings.position_pct)
        res.update({
            "n_windows": len(wf.windows),
            "avg_is_sharpe": float(wf.avg_is_sharpe),
            "avg_oos_sharpe": float(wf.avg_oos_sharpe),
            "oos_pos_frac": float(wf.positive_windows_frac),
            "n_trades_oos": int(sum(w.n_trades for w in wf.windows)),
            "degradation": float(wf.degradation),
        })

        # чутливість головного параметра (якщо є param_space)
        smoothness = float("nan")
        sens_n = 0
        ps = getattr(strategy, "param_space", {}) or {}
        if ps:
            pname = next(iter(ps))
            lo, hi, step = ps[pname]
            step = float(step) if step else 1.0
            n_vals = min(int((float(hi) - float(lo)) / step) + 1, 15)
            values = [float(lo) + i * step for i in range(max(n_vals, 2))][:15]
            try:
                sres = parameter_sensitivity(df, strategy, pname, values, cost=cost, funding=funding)
                smoothness = float(sres.smoothness)
                sens_n = len(sres.grid)
                res["sens_param"] = pname
            except Exception as exc:  # noqa: BLE001
                res["sens_error"] = str(exc)[:120]

        # Deflated Sharpe (як у CLI overfit: trials = combos × 50)
        res_full = run_backtest(df, strategy, cost=cost, funding=funding, position_pct=settings.position_pct)
        m = res_full.metrics
        res.update({
            "bt_total_return": float(m.total_return),
            "bt_sharpe": float(m.sharpe),
            "bt_max_dd": float(m.max_drawdown),
            "bt_n_trades": int(m.n_trades),
            "bt_profit_factor": float(m.profit_factor),
            "bt_win_rate": float(m.win_rate),
            "bt_trades_per_day": float(m.trades_per_day),
        })
        equity = res_full.equity
        ret = equity.pct_change().dropna()
        if len(ret) >= 2:
            combos = 1
            for _lo, _hi, _s in ps.values():
                _s = float(_s) if _s else 1.0
                combos *= max(int((float(_hi) - float(_lo)) / _s) + 1, 1)
            combos = min(max(combos, 1), 100_000)
            n_trials = estimate_n_trials(param_combinations=combos, backtests_per_combo=50)
            dsr = deflated_sharpe_ratio(ret.values, n_trials=n_trials)
            res["dsr"] = float(dsr)
            res["n_trials_dsr"] = int(n_trials)

        res["smoothness"] = float(smoothness) if smoothness == smoothness else None
        res["sens_n"] = sens_n
        res["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return res


def stage_audit() -> pd.DataFrame:
    winners = pd.read_csv(OUT_DIR / "winners_per_cell.csv")
    tasks = [
        {"symbol": r.symbol, "interval": r.interval, "strategy": r.winner_strategy}
        for r in winners.itertuples(index=False)
    ]
    print(f"Аудит {len(tasks)} комірок (ProcessPool, 6 воркерів)...")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_cell_audit, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{i}/{len(tasks)}] {r.get('symbol')} {r.get('interval')} "
                  f"{r.get('strategy')} status={r.get('status')}", flush=True)
    out = pd.DataFrame(rows)
    cols = ["symbol", "interval", "strategy", "status", "error", "n_windows",
            "avg_is_sharpe", "avg_oos_sharpe", "oos_pos_frac", "degradation",
            "n_trades_oos", "sens_param", "smoothness", "sens_n", "dsr",
            "n_trials_dsr", "bt_total_return", "bt_sharpe", "bt_max_dd",
            "bt_n_trades", "bt_profit_factor", "bt_win_rate", "bt_trades_per_day"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT_DIR / "audits.csv", index=False)
    print(out.to_markdown(index=False))
    print(f"\nЗбережено audits.csv ({len(out)} рядків)")
    return out


def verdict(row: pd.Series) -> tuple[str, str]:
    """(PASS/FAIL, причина) за критеріями overfitting-audit."""
    reasons = []
    oos = row.get("avg_oos_sharpe")
    if oos is None or pd.isna(oos) or oos <= 0.3:
        reasons.append(f"avg_oos_sharpe={oos:.3f}≤0.3")
    frac = row.get("oos_pos_frac")
    if frac is None or pd.isna(frac) or frac < 0.5:
        reasons.append(f"oos_pos_frac={frac:.0%}<50%")
    dsr = row.get("dsr")
    if dsr is None or pd.isna(dsr) or dsr <= 0.95:
        reasons.append(f"DSR={dsr if dsr is not None else float('nan'):.2f}≤0.95")
    sm = row.get("smoothness")
    if sm is None or pd.isna(sm) or sm <= 0.30:
        reasons.append(f"smoothness={sm if sm is not None else float('nan'):.2f}≤0.30")
    nt = row.get("bt_n_trades")
    min_nt = MIN_TRADES.get(row.get("interval"), 40)
    if nt is None or pd.isna(nt) or int(nt) < min_nt:
        reasons.append(f"n_trades={nt}<{min_nt}")
    if not reasons:
        return "PASS", ""
    return "FAIL", "; ".join(reasons)


def stage_report() -> str:
    """Фінальний markdown-звіт у docs/reports/."""
    import numpy as np
    bt = load_sweep("backtest")
    wf = load_sweep("walkforward")
    winners = pd.read_csv(OUT_DIR / "winners_per_cell.csv")
    audits = pd.read_csv(OUT_DIR / "audits.csv") if (OUT_DIR / "audits.csv").exists() else pd.DataFrame()

    lines = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Повна матриця дослідження стратегій — {ts}")
    lines.append("")
    lines.append("Параметри: 180 днів (2026-03-08 → 2026-09-04), 15 символів USDT-M, 6 таймфреймів, "
                 "9 швидких single-symbol стратегій, комісії maker/taker Binance + slippage 2 bps. "
                 "Старші ТФ ресемплені з 1m-кешу (Postgres, mainnet).")
    lines.append("")
    ok_bt = bt[bt["status"] == "ok"]
    ok_wf = wf[wf["status"] == "ok"]
    lines.append(f"- клітинок backtest: {len(bt)} (ok: {len(ok_bt)})")
    lines.append(f"- клітинок walkforward: {len(wf)} (ok: {len(ok_wf)})")
    if not audits.empty:
        npass = audits.apply(verdict, axis=1, result_type="expand")[0].value_counts().to_dict()
        lines.append(f"- аудитів комірок: {len(audits)} (PASS: {npass.get('PASS', 0)}, FAIL: {npass.get('FAIL', 0)})")
    lines.append("")
    if not ok_bt.empty:
        lines.append("## Загальний розподіл: найкращі стратегії за середнім Sharpe (backtest)")
        lines.append("")
        agg = (ok_bt.groupby("strategy")
               .agg(mean_sharpe=("sharpe", "mean"), med_sharpe=("sharpe", "median"),
                    max_sharpe=("sharpe", "max"), cells_ok=("sharpe", "count"),
                    cells_positive=("sharpe", lambda s: int((s > 0).sum())))
               .sort_values("mean_sharpe", ascending=False).reset_index())
        for c in ["mean_sharpe", "med_sharpe", "max_sharpe"]:
            agg[c] = agg[c].map(lambda v: f"{v:+.2f}")
        lines.append(agg.to_markdown(index=False))
        lines.append("")
    if not ok_wf.empty:
        lines.append("## Walk-forward OOS: стратегії за середнім OOS Sharpe")
        lines.append("")
        agg = (ok_wf.groupby("strategy")
               .agg(mean_oos=("avg_oos_sharpe", "mean"), med_oos=("avg_oos_sharpe", "median"),
                    cells_ok=("avg_oos_sharpe", "count"),
                    cells_oos_pos=("avg_oos_sharpe", lambda s: int((s > 0).sum())))
               .sort_values("mean_oos", ascending=False).reset_index())
        for c in ["mean_oos", "med_oos"]:
            agg[c] = agg[c].map(lambda v: f"{v:+.3f}")
        lines.append(agg.to_markdown(index=False))
        lines.append("")
    if not ok_bt.empty and not ok_wf.empty:
        lines.append("## Розподіл по таймфреймах (середній OOS Sharpe по стратегіях)")
        lines.append("")
        pivot = ok_wf.pivot_table(index="interval", columns="strategy", values="avg_oos_sharpe", aggfunc="mean")
        lines.append(pivot.round(3).to_markdown())
        lines.append("")

    lines.append("## Переможець на кожну комірку (символ × ТФ) за OOS")
    lines.append("")
    lines.append("| символ | ТФ | стратегія | OOS Sharpe | OOS>0 | IS Sharpe | кандидатів |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in winners.sort_values(["interval", "symbol"]).iterrows():
        lines.append(f"| {r['symbol']} | {r['interval']} | {r['winner_strategy']} | "
                     f"{r['winner_avg_oos_sharpe']:+.3f} | {r['winner_oos_pos_frac']:.0%} | "
                     f"{r['winner_avg_is_sharpe']:+.3f} | {r['n_candidates_ok']} |")
    lines.append("")

    if not audits.empty:
        lines.append("## Аудит переможців (WF + чутливість + Deflated Sharpe)")
        lines.append("")
        lines.append("| символ | ТФ | стратегія | вердикт | avg_OOS | OOS>0 | DSR | smooth | trades | ret | Sharpe | причина |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in audits.sort_values(["interval", "symbol"]).iterrows():
            v, why = verdict(r)
            oos = r.get("avg_oos_sharpe")
            frac = r.get("oos_pos_frac")
            dsr = r.get("dsr")
            sm = r.get("smoothness")
            nt = r.get("bt_n_trades")
            ret = r.get("bt_total_return")
            sh = r.get("bt_sharpe")
            fmt = lambda x, d=2, suf="": (f"{x:+.3f}" if suf == "s" else f"{x:+.2%}" if suf == "pct" else
                                          (f"{x:.0f}" if x is not None and pd.notna(x) else "-"))
            lines.append(
                f"| {r['symbol']} | {r['interval']} | {r['strategy']} | **{v}** | "
                f"{fmt(oos, suf='s')} | {fmt(frac, suf='pct') if frac is not None and pd.notna(frac) else '-'} | "
                f"{fmt(dsr)} | {fmt(sm)} | {fmt(nt)} | "
                f"{fmt(ret, suf='pct') if ret is not None and pd.notna(ret) else '-'} | "
                f"{fmt(sh, suf='s')} | {why or ''} |")
        lines.append("")

    lines.append("## Критерії PASS")
    lines.append("avg OOS Sharpe > 0.30; частка OOS-вікон > 0 ≥ 50%; DSR > 0.95; smoothness > 0.30; "
                 "кількість угод ≥ порогу ТФ. Висновок про edge — лише після PASS та холдауту.")
    report = "\n".join(lines)
    out = ROOT / "docs" / "reports" / f"full_matrix_{datetime.datetime.now():%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nЗвіт: {out}")
    return str(out)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "winners"
    if stage == "winners":
        stage_winners()
    elif stage == "audit":
        stage_audit()
    elif stage == "report":
        stage_report()
    else:
        print("невідомий stage", stage)
        sys.exit(2)
