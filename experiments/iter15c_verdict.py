"""RS-вердикт (iter15c) — чесний вердикт із cached sleeves: вибір переможця ЛИШЕ на selection half.

Читає results/iter15/sleeves/{interval}_{mode}_{symbol}.parquet, проганяє зареєстровані
політики (з каузальним лагом на рішенні і з відніманням вартості re-weighting),
вибирає переможця за selection half і рахує pre-registered гейт на validation half.

Приклади:
    UV_CACHE_DIR=$PWD/.uvcache uv run python experiments/iter15c_verdict.py --interval 1d --cost-mode maker
    ... --symbols APTUSDT,ARBUSDT --out-tag fresh --interval 4h \
        --detectors det_rule,det_vol,det_mkt,det_btcvol \
        --policies argmax,gap_dwell,soft_shrink,riskoff_anchor,riskoff_gate --n-trials 85
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
N_TRIALS_DEFAULT = 63  # RS-1 pre-registered


def load_harness():
    spec = importlib.util.spec_from_file_location("it15", ROOT / "experiments" / "iter15_regime_supervisor_cycle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--cost-mode", default="maker")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--detectors", default="det_rule,det_vol,det_mkt")
    ap.add_argument("--policies", default="argmax,gap_dwell,soft_shrink,riskoff_anchor")
    ap.add_argument("--off-dwell", type=int, default=6)
    ap.add_argument("--on-dwell", type=int, default=12)
    ap.add_argument("--risk-off-scale", type=float, default=0.25)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    args = ap.parse_args()

    h = load_harness()
    ppy = h.PPY[args.interval]
    days = h.DEFAULT_DAYS[args.interval]
    cost_side = 0.0004 if args.cost_mode == "maker" else 0.0007  # fee + slippage, як у CostModel
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or list(h.CORE15)
    det_list = [d.strip() for d in args.detectors.split(",") if d.strip()]
    pol_list = [p.strip() for p in args.policies.split(",") if p.strip()]
    ref = f"{args.interval}_{args.cost_mode}" + (f"_{args.out_tag}" if args.out_tag else "")

    mkt = h.market_states(args.interval, max(days, 2500))
    btc_vol = h.btc_vol_states(args.interval, max(days, 2500))

    panels: dict[str, dict[str, pd.Series]] = {}
    extra: dict[str, dict[str, dict]] = {}
    for symbol in symbols:
        p = SLEEVE_DIR / f"{args.interval}_{args.cost_mode}_{symbol}.parquet"
        if not p.exists():
            print(f"skip {symbol}: немає sleeves")
            continue
        R = pd.read_parquet(p)
        df = h.load_klines(symbol, args.interval, days)
        st = {d: h.symbol_regimes(df, d, mkt, btc_vol).reindex(R.index).shift(1) for d in det_list}
        variants: list[tuple[str, str]] = [
            ("incumbent", "-"),
            ("equal", "-"),
            ("best_single_train", "-"),
            ("oracle", "-"),
        ]
        for d in det_list:
            for pol in pol_list:
                variants.append((pol, d))
        for pol, det in variants:
            k = pol if det == "-" else f"{pol}@{det}"
            sd = st[det] if det != "-" else st[det_list[0]]
            res = h.run_policy(
                R,
                sd,
                pol,
                h.compute_folds(R.index),
                ppy,
                cost_per_side=cost_side,
                off_dwell=args.off_dwell,
                on_dwell=args.on_dwell,
                risk_off_scale=args.risk_off_scale,
            )
            panels.setdefault(k, {})[symbol] = res["pnl"]
            extra.setdefault(k, {})[symbol] = {"switches": res["switch_bars"], "turnover": res["turnover"]}
    if not panels:
        raise SystemExit(f"немає збережених sleeves для {ref}")

    common = None
    for d in panels.values():
        idx = pd.DataFrame(d).sort_index().index
        common = idx if common is None else common.union(idx)
    common = common.sort_values()
    split = len(common) // 2
    sel_idx, val_idx = common[:split], common[split:]

    def port(d: dict[str, pd.Series]) -> pd.Series:
        return pd.DataFrame(d).sort_index().reindex(common).mean(axis=1).fillna(0.0)

    rows: list[dict] = []
    for k, d in panels.items():
        sym_sel = [h.sharpe(s.reindex(sel_idx).to_numpy(), ppy, min_bars=60) for s in d.values()]
        sym_val = [h.sharpe(s.reindex(val_idx).to_numpy(), ppy, min_bars=60) for s in d.values()]
        sym_sel = [x for x in sym_sel if np.isfinite(x)]
        sym_val = [x for x in sym_val if np.isfinite(x)]
        pf = port(d)
        pv = pf.reindex(val_idx)
        ci_lo, ci_hi = h.bootstrap_sharpe_ci(pv.to_numpy(), ppy)
        dd = h.max_dd(pv)
        rows.append(
            {
                "variant": k,
                "mean_sym_sr_sel": float(np.mean(sym_sel)) if sym_sel else float("nan"),
                "mean_sym_sr_val": float(np.mean(sym_val)) if sym_val else float("nan"),
                "port_sr_sel": h.sharpe(pf.reindex(sel_idx).to_numpy(), ppy, min_bars=60),
                "port_sr_val": h.sharpe(pv.to_numpy(), ppy, min_bars=60),
                "t_nw_val": h.newey_west_t(pv.to_numpy()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "frac_pos_val": float(np.mean([x > 0 for x in sym_val])) if sym_val else float("nan"),
                "maxdd_val": dd,
                "calmar_val": float(pv.mean() * ppy / abs(dd)) if dd < 0 else 0.0,
                "switches_total": int(sum(extra.get(k, {}).get(s, {}).get("switches", 0) for s in d)),
                "turnover": float(np.mean([extra.get(k, {}).get(s, {}).get("turnover", np.nan) for s in d])),
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"verdict_table_{ref}.csv", index=False)

    from scalper_hft.validation.cscv import pbo_cscv
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio

    mat = pd.DataFrame({k: port(d) for k, d in panels.items() if k != "oracle"}).dropna()
    pbo = float(pbo_cscv(mat.to_numpy().T, n_blocks=8, purge_bars=2, embargo_bars=2).pbo)

    selectable = res[res["variant"] != "oracle"].sort_values("mean_sym_sr_sel", ascending=False)
    winner = str(selectable.iloc[0]["variant"])
    w = res[res["variant"] == winner].iloc[0]
    pv = port(panels[winner]).reindex(val_idx)
    dsr = float(deflated_sharpe_ratio(pv.to_numpy(), n_trials=args.n_trials))
    inc = res[res["variant"] == "incumbent"].iloc[0]
    bst = res[res["variant"] == "best_single_train"].iloc[0]
    eq = res[res["variant"] == "equal"].iloc[0]

    gate = {
        "ref": ref,
        "interval": args.interval,
        "cost_mode": args.cost_mode,
        "n_symbols": len(panels[winner]),
        "winner_by_selection_half": winner,
        "G1_mean_sym_sr_val_ge_0.5": bool(w["mean_sym_sr_val"] >= 0.5),
        "G2_frac_pos_ge_0.7": bool(w["frac_pos_val"] >= 0.7),
        "G3_delta_vs_incumbent_ge_0.2": bool(w["port_sr_val"] - inc["port_sr_val"] >= 0.2),
        "G4_delta_vs_best_single_ge_0.2": bool(w["port_sr_val"] - bst["port_sr_val"] >= 0.2),
        "G4b_delta_vs_equal_ge_0.2": bool(w["port_sr_val"] - eq["port_sr_val"] >= 0.2),
        "G5_t_nw_ge_2": bool(w["t_nw_val"] >= 2.0),
        "G5b_ci_excludes_zero": bool(w["ci_lo"] > 0),
        "G6_pbo_lt_0.25": bool(pbo < 0.25),
        "G7_dsr_ge_0.95": bool(dsr >= 0.95),
        "metrics": {
            "port_sr_val": float(w["port_sr_val"]),
            "mean_sym_sr_val": float(w["mean_sym_sr_val"]),
            "t_nw_val": float(w["t_nw_val"]),
            "ci": [float(w["ci_lo"]), float(w["ci_hi"])],
            "maxdd_val": float(w["maxdd_val"]),
            "calmar_val": float(w["calmar_val"]),
            "pbo": pbo,
            "dsr": dsr,
            "n_trials": int(args.n_trials),
            "incumbent_port_sr_val": float(inc["port_sr_val"]),
            "incumbent_calmar_val": float(inc["calmar_val"]),
            "incumbent_maxdd_val": float(inc["maxdd_val"]),
            "best_single_port_sr_val": float(bst["port_sr_val"]),
            "equal_port_sr_val": float(eq["port_sr_val"]),
            "oracle_port_sr_val": float(res[res["variant"] == "oracle"].iloc[0]["port_sr_val"]),
        },
    }
    tier1 = all(v for k, v in gate.items() if k.startswith("G") and isinstance(v, bool))
    gate["tier1_pass"] = bool(tier1)
    gate["tier2_partial"] = bool(
        (w["port_sr_val"] - inc["port_sr_val"]) >= -0.05
        and (w["calmar_val"] - inc["calmar_val"]) >= 0.20
        and (w["maxdd_val"] - inc["maxdd_val"]) <= 0.0
    )
    (OUT / f"gate_{ref}.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Вердикт · {ref}",
        "",
        f"Переможець за **selection half**: `{winner}` · символів: {len(panels[winner])}",
        "",
        res.round(3).to_markdown(index=False),
        "",
        "## Гейт (pre-registered)",
        "",
        "```json",
        json.dumps(gate, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (OUT / f"verdict_{ref}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
