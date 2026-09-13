"""iter17 (RS-3) — верифікація: стратегія `RegimeSupervisor(blend_mode="risk_overlay")` через СВІЙ інтерфейс.

Питання (pre-registration hypothesis_iter17_rs3.md §3): чи відтворює реалізований у
стратегії механізм ті самі числа, що дослідницький harness
(`results/iter15/verdict_table_4h_maker_fresh.csv`, політика `riskoff_anchor@det_vol`)?

Метод: walk-forward (train=250/test=125, purge=2, embargo=2, maker) на 4h для
    (а) supervisor(blend_mode="risk_overlay") — заморожений пул 5 рукавів;
    (б) incumbent окремо (ts_momentum long-only) як контроль.
Далі порівняння з harness-числами: Sharpe портфеля, кореляція per-symbol PnL.

Запуск:
    UV_CACHE_DIR=$PWD/.uvcache uv run python experiments/iter17_risk_overlay_verify.py \
        --symbols APTUSDT,ARBUSDT,... --interval 4h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter17"
FRESH = [
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "FILUSDT",
    "ETCUSDT", "TRXUSDT", "ALGOUSDT", "RUNEUSDT", "SANDUSDT",
]


def load_harness():
    import importlib.util

    spec = importlib.util.spec_from_file_location("it15", ROOT / "experiments" / "iter15_regime_supervisor_cycle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--symbols", default=",".join(FRESH))
    ap.add_argument("--train", type=int, default=250)
    ap.add_argument("--test", type=int, default=125)
    ap.add_argument("--days", type=int, default=2500)
    args = ap.parse_args()

    h = load_harness()
    OUT.mkdir(parents=True, exist_ok=True)
    ppy = h.PPY[args.interval]

    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.store import get_store
    from scalper_hft.strategies import get_strategy
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor
    from scalper_hft.validation.walk_forward import run_walk_forward

    cost = CostModel.from_settings(get_settings())
    store = get_store()
    rows: list[dict] = []
    pnl_sup: dict[str, pd.Series] = {}
    pnl_inc: dict[str, pd.Series] = {}

    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        df = h.load_klines(symbol, args.interval, args.days)
        funding = store.load_funding(symbol)
        sup = RegimeSupervisor(blend_mode="risk_overlay", hmm_fit_bars=2000)
        wf_sup = run_walk_forward(
            df, sup, train_bars=args.train, test_bars=args.test, cost=cost,
            funding=funding, is_maker=True, interval=args.interval,
            collect_oos_returns=True, purge_bars=2, embargo_bars=2, strict_data=False,
        )
        inc = get_strategy("ts_momentum", lookback=20, allow_short=False)
        wf_inc = run_walk_forward(
            df, inc, train_bars=args.train, test_bars=args.test, cost=cost,
            is_maker=True, interval=args.interval,
            collect_oos_returns=True, purge_bars=2, embargo_bars=2, strict_data=False,
        )
        a = wf_sup.oos_returns
        a = a[~a.index.duplicated(keep="first")]
        b = wf_inc.oos_returns
        b = b[~b.index.duplicated(keep="first")]
        pnl_sup[symbol] = a
        pnl_inc[symbol] = b
        # діагностика: частка risk-off барів (детектор стратегії vs harness-детектор)
        from scalper_hft.features.regimes import volatility_regime

        det = sup._detector.detect(df["close"])
        ro_strategy = float((det["vol"].reindex(a.index).fillna("normal") == "high").mean())
        ro_harness = float((volatility_regime(df["close"]).reindex(a.index).fillna("normal") == "high").mean())
        rows.append(
            {
                "symbol": symbol,
                "n_bars": len(a),
                "risk_off_frac_strategy": ro_strategy,
                "risk_off_frac_harness": ro_harness,
                "sup_sharpe": h.sharpe(a.to_numpy(), ppy, min_bars=60),
                "inc_sharpe": h.sharpe(b.to_numpy(), ppy, min_bars=60),
                "sup_trades": int(sum(w.n_trades for w in wf_sup.windows)),
                "inc_trades": int(sum(w.n_trades for w in wf_inc.windows)),
            }
        )
        print(f"{symbol:<9s} sup SR {rows[-1]['sup_sharpe']:+.3f} | inc SR {rows[-1]['inc_sharpe']:+.3f}", flush=True)

    per_sym = pd.DataFrame(rows)
    per_sym.to_csv(OUT / f"verify_per_symbol_{args.interval}.csv", index=False)
    # серії PnL по символах (для порівняння з harness)
    pd.concat({"supervisor": pd.DataFrame(pnl_sup), "incumbent": pd.DataFrame(pnl_inc)}, axis=1).to_parquet(
        OUT / f"pnl_{args.interval}.parquet"
    )

    def portfolio(d: dict[str, pd.Series]) -> pd.Series:
        return pd.DataFrame(d).sort_index().mean(axis=1).dropna()

    port_sup, port_inc = portfolio(pnl_sup), portfolio(pnl_inc)
    split = len(port_sup) // 2
    summary: dict = {"interval": args.interval, "n_symbols": len(per_sym)}
    for name, s in (("supervisor_risk_overlay", port_sup), ("incumbent", port_inc)):
        v = s.iloc[split:]
        lo, hi = h.bootstrap_sharpe_ci(v.to_numpy(), ppy)
        dd = h.max_dd(v)
        summary[name] = {
            "port_sr_val": h.sharpe(v.to_numpy(), ppy, min_bars=60),
            "t_nw_val": h.newey_west_t(v.to_numpy()),
            "ci": [lo, hi],
            "maxdd_val": dd,
            "calmar_val": float(v.mean() * ppy / abs(dd)) if dd < 0 else 0.0,
            "frac_sym_pos_val": float(
                (
                    (per_sym["sup_sharpe"] > 0) if name.startswith("supervisor") else (per_sym["inc_sharpe"] > 0)
                ).mean()
            ),
        }
    summary["mean_sym_sr_val"] = {
        "supervisor": float(per_sym["sup_sharpe"].mean()),
        "incumbent": float(per_sym["inc_sharpe"].mean()),
    }
    summary["delta_sr_vs_incumbent"] = float(
        summary["supervisor_risk_overlay"]["port_sr_val"] - summary["incumbent"]["port_sr_val"]
    )

    # ── порівняння з harness (RS-2, 4h maker fresh) ──
    cmp_rows = []
    verdict = ROOT / "results" / "iter15" / f"verdict_table_{args.interval}_maker_fresh.csv"
    if verdict.exists():
        vt = pd.read_csv(verdict).set_index("variant")
        for variant, key in (("riskoff_anchor@det_vol", "supervisor_risk_overlay"), ("incumbent", "incumbent")):
            if variant not in vt.index:
                continue
            cmp_rows.append(
                {
                    "variant": variant,
                    "harness_val_sr": float(vt.loc[variant, "port_sr_val"]),
                    "strategy_val_sr": float(summary[key]["port_sr_val"]),
                    "delta": float(summary[key]["port_sr_val"] - vt.loc[variant, "port_sr_val"]),
                    "harness_t_nw": float(vt.loc[variant, "t_nw_val"]),
                    "strategy_t_nw": float(summary[key]["t_nw_val"]),
                }
            )
    summary["harness_reference"] = cmp_rows
    (OUT / f"verify_{args.interval}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# RS-3 верифікація risk_overlay · {args.interval}",
        "",
        f"Символів: {len(per_sym)} · OOS барів: {len(port_sup)} ({port_sup.index[0]:%Y-%m-%d} → {port_sup.index[-1]:%Y-%m-%d})",
        "",
        per_sym.round(3).to_markdown(index=False),
        "",
        "## Портфель (validation half)",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (OUT / f"verify_{args.interval}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-14:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
