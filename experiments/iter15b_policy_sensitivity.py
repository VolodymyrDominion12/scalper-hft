"""iter15b (RS-1, аудит) — sensitivity гейта G8 + cross-symbol transfer (HB-11 proxy) + PBO по сім'ях.

Читає ЗБЕРЕЖЕНІ матриці рукавів (results/iter15/sleeves/*.parquet) — дорогий
walk-forward не перераховується, тому сітка параметрів і трансфер коштують секунди.

Запуск:
    UV_CACHE_DIR=$PWD/.uvcache uv run python experiments/iter15b_policy_sensitivity.py --interval 1d --cost-mode maker
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iter15"
SLEEVE_DIR = OUT / "sleeves"


def load_harness():
    spec = importlib.util.spec_from_file_location("it15", ROOT / "experiments" / "iter15_regime_supervisor_cycle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sleeves(interval: str, cost_mode: str, symbols: list[str], h) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        p = SLEEVE_DIR / f"{interval}_{cost_mode}_{symbol}.parquet"
        if p.exists():
            out[symbol] = pd.read_parquet(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--cost-mode", default="maker")
    ap.add_argument("--pool-train", type=int, default=10, help="скільки символів у train-пулі для transfer")
    args = ap.parse_args()

    h = load_harness()
    ppy = h.PPY[args.interval]
    days = h.DEFAULT_DAYS[args.interval]
    sleeves = load_sleeves(args.interval, args.cost_mode, h.CORE15, h)
    if not sleeves:
        raise SystemExit(f"немає збережених sleeves у {SLEEVE_DIR} для {args.interval}_{args.cost_mode}")
    mkt = h.market_states(args.interval, max(days, 2500))

    states: dict[str, dict[str, pd.Series]] = {}
    for symbol, R in sleeves.items():
        df = h.load_klines(symbol, args.interval, days)
        states[symbol] = {
            "det_rule": h.symbol_regimes(df, "det_rule", mkt).reindex(R.index),
            "det_vol": h.symbol_regimes(df, "det_vol", mkt).reindex(R.index),
            "det_mkt": h.symbol_regimes(df, "det_mkt", mkt).reindex(R.index),
        }
        states[symbol] = {k: v.shift(1) for k, v in states[symbol].items()}
    print(f"sleeves: {len(sleeves)} символів")

    def evaluate(policy: str, det: str, pool: dict[str, pd.DataFrame] | None = None, **kw) -> dict:
        """Portfolio Sharpe на validation half + per-symbol метрики."""
        port, sym_sr, switch = {}, {}, {}
        for symbol, R in sleeves.items():
            p = None
            if pool is not None:
                p = [(pool[s], states[s][det]) for s in pool if s != symbol]
            res = h.run_policy(R, states[symbol][det], policy, h.compute_folds(R.index), ppy, pool=p, **kw)
            port[symbol] = res["pnl"]
            switch[symbol] = res["switch_bars"]
        mat = pd.DataFrame(port).sort_index()
        pf = mat.mean(axis=1)
        split = len(pf) // 2
        val = pf.iloc[split:]
        for symbol, s in port.items():
            sym_sr[symbol] = h.sharpe(s.reindex(val.index).to_numpy(), ppy, min_bars=60)
        finite = [v for v in sym_sr.values() if np.isfinite(v)]
        return {
            "port_sr_val": h.sharpe(val.to_numpy(), ppy, min_bars=60),
            "port_sr_sel": h.sharpe(pf.iloc[:split].to_numpy(), ppy, min_bars=60),
            "t_nw_val": h.newey_west_t(val.to_numpy()),
            "mean_sym_sr_val": float(np.mean(finite)) if finite else float("nan"),
            "frac_pos_val": float(np.mean([v > 0 for v in finite])) if finite else float("nan"),
            "maxdd_val": h.max_dd(val),
            "switches_total": int(sum(switch.values())),
        }

    # ── 1. Sensitivity сітки (G8): gap × dwell × detector ──
    rows = []
    for det in h.DETECTORS:
        for gap in (0.3, 0.4, 0.5, 0.6, 0.75, 1.0):
            for dwell in (3, 5, 8, 12):
                m = evaluate("gap_dwell", det, gap=gap, dwell=dwell)
                rows.append({"family": "gap_dwell", "detector": det, "gap": gap, "dwell": dwell, **m})
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / f"sensitivity_{args.interval}_{args.cost_mode}.csv", index=False)

    # ── 2. Sensitivity soft_shrink (tau × shrink) ──
    rows2 = []
    for det in h.DETECTORS:
        for tau in (0.5, 1.0, 2.0):
            for shrink in (0.3, 0.5, 0.7):
                m = evaluate("soft_shrink", det, tau=tau, shrink=shrink)
                rows2.append({"family": "soft_shrink", "detector": det, "tau": tau, "shrink": shrink, **m})
    sens2 = pd.DataFrame(rows2)
    sens2.to_csv(OUT / f"sensitivity_soft_{args.interval}_{args.cost_mode}.csv", index=False)

    # ── 3. Cross-symbol transfer (HB-11 proxy): train map на пулі, тест на held-out ──
    syms = list(sleeves.keys())
    train_pool = syms[: args.pool_train]
    held_out = syms[args.pool_train :]
    rows3 = []
    if held_out:
        for det in h.DETECTORS:
            for pol in ("argmax", "gap_dwell"):
                # per-symbol (як у головному прогоні) — лише на held-out
                port_per, port_pool, sw = {}, {}, {}
                for symbol in held_out:
                    R = sleeves[symbol]
                    res_per = h.run_policy(R, states[symbol][det], pol, h.compute_folds(R.index), ppy)
                    pool_frames = [(sleeves[s], states[s][det]) for s in train_pool]
                    res_pool = h.run_policy(
                        R, states[symbol][det], pol, h.compute_folds(R.index), ppy, pool=pool_frames
                    )
                    port_per[symbol] = res_per["pnl"]
                    port_pool[symbol] = res_pool["pnl"]
                    sw[symbol] = (res_per["switch_bars"], res_pool["switch_bars"])
                for label, dd in (("per_symbol", port_per), ("pooled_map", port_pool)):
                    mat = pd.DataFrame(dd).sort_index()
                    pf = mat.mean(axis=1)
                    split = len(pf) // 2
                    val = pf.iloc[split:]
                    sr = [h.sharpe(dd[s].reindex(val.index).to_numpy(), ppy, min_bars=60) for s in held_out]
                    sr = [x for x in sr if np.isfinite(x)]
                    rows3.append(
                        {
                            "detector": det,
                            "policy": pol,
                            "fit": label,
                            "n_held_out": len(held_out),
                            "port_sr_val": h.sharpe(val.to_numpy(), ppy, min_bars=60),
                            "mean_sym_sr_val": float(np.mean(sr)) if sr else float("nan"),
                            "frac_pos_val": float(np.mean([x > 0 for x in sr])) if sr else float("nan"),
                        }
                    )
    trans = pd.DataFrame(rows3)
    if not trans.empty:
        trans.to_csv(OUT / f"transfer_{args.interval}_{args.cost_mode}.csv", index=False)

    # ── 4. PBO по сім'ях політик ──
    pbo_rows = []
    from scalper_hft.validation.cscv import pbo_cscv

    port_file = OUT / f"portfolio_returns_{args.interval}_{args.cost_mode}.parquet"
    if port_file.exists():
        port_df = pd.read_parquet(port_file)
        families = {
            "all": [c for c in port_df.columns if c != "oracle"],
            "argmax": [c for c in port_df.columns if c.startswith("argmax")],
            "gap_dwell": [c for c in port_df.columns if c.startswith("gap_dwell")],
            "soft_shrink": [c for c in port_df.columns if c.startswith("soft_shrink")],
            "riskoff_anchor": [c for c in port_df.columns if c.startswith("riskoff_anchor")],
        }
        for name, cols in families.items():
            if len(cols) < 3:
                continue
            m = port_df[cols].dropna()
            res = pbo_cscv(m.to_numpy().T, n_blocks=8, purge_bars=2, embargo_bars=2)
            pbo_rows.append({"family": name, "n_variants": len(cols), "pbo": float(res.pbo)})
    pbo_df = pd.DataFrame(pbo_rows)
    if not pbo_df.empty:
        pbo_df.to_csv(OUT / f"pbo_families_{args.interval}_{args.cost_mode}.csv", index=False)

    # ── 5. Звіт ──
    lines = [f"# iter15b — sensitivity / transfer / PBO ({args.interval}, {args.cost_mode})", ""]
    lines.append("## G8: сітка gap × dwell (gap_dwell)")
    lines.append("")
    piv = sens.pivot_table(index=["detector", "gap"], columns="dwell", values="port_sr_val")
    lines.append(piv.round(3).to_markdown())
    lines.append("")
    lines.append("## G8: сітка tau × shrink (soft_shrink)")
    lines.append("")
    piv2 = sens2.pivot_table(index=["detector", "tau"], columns="shrink", values="port_sr_val")
    lines.append(piv2.round(3).to_markdown())
    lines.append("")
    if not trans.empty:
        lines.append("## HB-11 proxy: cross-symbol transfer (train на перших символах → held-out)")
        lines.append("")
        lines.append(trans.round(3).to_markdown(index=False))
        lines.append("")
    if not pbo_df.empty:
        lines.append("## PBO по сім'ях політик")
        lines.append("")
        lines.append(pbo_df.round(3).to_markdown(index=False))
        lines.append("")
    best = sens.sort_values("port_sr_val", ascending=False).head(5)
    lines.append("## Топ-5 клітинок сітки gap_dwell")
    lines.append("")
    lines.append(best.round(3).to_markdown(index=False))
    (OUT / f"sensitivity_report_{args.interval}_{args.cost_mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
