"""Iteration 7 — аналіз: per-regime атрибуція + чесний тест regime-switching.

Вхід:
    results/iter7_shard_*.csv        — метрики клітинок (з iter7_regime_rating.py)
    results/iter7_oos/<sym>__<v>.parquet — OOS бар-дохідності + режимна мітка

Вихід:
    results/iter7_rating.csv         — агрегований рейтинг варіантів
    results/iter7_regime_matrix.csv  — Sharpe per (variant × regime × symbol)
    results/iter7_switch_test.csv    — тест «вибір стратегії за режимом» на H1→H2
    results/iter7_switch_summary.csv — те саме, агреговано по символах

Метод тесту перемикання (без lookahead і без підгонки):
    OOS-дохідності walk-forward діляться навпіл за часом: H1 (перша половина
    OOS-барів) і H2 (друга). Селектор режим→стратегія «навчається» ЛИШЕ на H1
    (для кожного режиму — стратегія з найвищим Sharpe на H1; якщо найкращий
    Sharpe ≤ 0, режим у H2 не торгується). Далі ця карта застосовується до H2.
    Порівнюємо з (a) найкращою одиночною стратегією, обраною на H1 і
    застосованою до всього H2 (без перемикання — головний конкурент);
    (b) рівноважним усередненням усіх singles; (c) oracle на H2 (ex-post межа).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PPY = 8760.0
OOS_DIR = Path("results/iter7_oos")
RATING_CSV = Path("results/iter7_rating.csv")
REGIME_CSV = Path("results/iter7_regime_matrix.csv")
SWITCH_CSV = Path("results/iter7_switch_test.csv")
SWITCH_SUM_CSV = Path("results/iter7_switch_summary.csv")

REGIMES = [("range", "флет"), ("trend_up", "бичий"), ("trend_down", "ведмежий")]
MIN_BARS_SELECT = 100  # мінімум барів режиму на H1, щоб обирати стратегію


def sharpe(r: pd.Series, min_bars: int = 30) -> float:
    r = pd.Series(r).dropna()
    if len(r) < min_bars:
        return float("nan")
    sd = float(r.std(ddof=0))
    if sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(PPY))


def load_cells(csv_glob: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(Path().glob(csv_glob))]
    if not frames:
        raise SystemExit(f"немає CSV за шаблоном {csv_glob}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["symbol", "variant"], keep="first")


def load_oos(symbol: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Повертає (матриця дохідностей варіантів, мітка режиму, структура)."""
    series: dict[str, pd.Series] = {}
    labels: pd.Series | None = None
    structure: pd.Series | None = None
    for p in sorted(OOS_DIR.glob(f"{symbol}__*.parquet")):
        label = p.stem.split("__", 1)[1].replace("_", ":", 1)
        f = pd.read_parquet(p)
        f = f[~f.index.duplicated(keep="first")]
        series[label] = f["ret"].astype(float)
        if labels is None:
            labels = f["structure"].astype(str)
            structure = f["vol"].astype(str) if "vol" in f.columns else None
    if not series:
        raise SystemExit(f"немає OOS-файлів для {symbol}")
    mat = pd.DataFrame(series).sort_index()
    lab = labels.reindex(mat.index).fillna("range")
    return mat, lab, structure if structure is not None else pd.Series("normal", index=mat.index)


def rate(cells: pd.DataFrame) -> pd.DataFrame:
    ok = cells[cells["status"] == "ok"].copy()
    ok["is_pos"] = (ok["avg_oos_sharpe"] > 0).astype(float)
    ok["pass_gate"] = ((ok["avg_oos_sharpe"] > 0.3) & (ok["oos_positive_frac"] >= 0.5)).astype(float)
    agg = (
        ok.groupby(["variant", "group", "adaptive"], dropna=False)
        .agg(
            n_symbols=("symbol", "count"),
            mean_oos_sharpe=("avg_oos_sharpe", "mean"),
            median_oos_sharpe=("avg_oos_sharpe", "median"),
            mean_pooled_sharpe=("pooled_oos_sharpe", "mean"),
            mean_pos_frac=("oos_positive_frac", "mean"),
            mean_ret=("oos_total_return", "mean"),
            worst_dd=("oos_max_dd", "min"),
            n_trades=("n_trades_oos", "sum"),
            sharpe_flat=("sharpe_range", "mean"),
            sharpe_bull=("sharpe_up", "mean"),
            sharpe_bear=("sharpe_down", "mean"),
            frac_symbols_positive=("is_pos", "mean"),
            frac_symbols_pass_gate=("pass_gate", "mean"),
        )
        .reset_index()
    )
    return agg.sort_values("mean_oos_sharpe", ascending=False).reset_index(drop=True)


def regime_matrix(oos_by_symbol: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, (mat, lab, _vol) in oos_by_symbol.items():
        for label in mat.columns:
            for key, _uk in REGIMES:
                sub = mat.loc[lab == key, label]
                if len(sub) < 30:
                    continue
                rows.append({"symbol": symbol, "variant": label, "regime": key, "sharpe": sharpe(sub, 30)})
    return pd.DataFrame(rows)


def switch_test(
    symbol: str,
    mat: pd.DataFrame,
    lab: pd.Series,
    pool: list[str],
) -> dict:
    """Селектор режим→стратегія, навчений на H1, застосований до H2."""
    cols = [c for c in pool if c in mat.columns]
    R = mat[cols]
    n = len(R)
    half = n // 2
    idx_h1, idx_h2 = R.index[:half], R.index[half:]
    lab_h1, lab_h2 = lab.reindex(idx_h1), lab.reindex(idx_h2)

    # ── селектор: найкраща стратегія per-regime на H1 ──────────────────────
    choice: dict[str, str | None] = {}
    for key, _uk in REGIMES:
        sub = R.loc[lab_h1 == key]
        best_lab, best_sh = None, -np.inf
        for c in cols:
            s = sub[c]
            if int(s.notna().sum()) < MIN_BARS_SELECT:
                continue
            sh = sharpe(s, MIN_BARS_SELECT)
            if np.isnan(sh):
                continue
            if sh > best_sh:
                best_lab, best_sh = c, sh
        choice[key] = best_lab if (best_lab is not None and best_sh > 0) else None

    # ── H2: перемикання за режимом ─────────────────────────────────────────
    switch_ret = pd.Series(0.0, index=idx_h2)
    for key, _uk in REGIMES:
        c = choice[key]
        if c is None:
            continue
        switch_ret.loc[lab_h2 == key] = R.loc[lab_h2 == key, c]

    # базова лінія: одна стратегія на весь H2, обрана на H1 (без перемикання)
    best_one, best_one_sh = None, -np.inf
    for c in cols:
        sh = sharpe(R.loc[idx_h1, c], 200)
        if not np.isnan(sh) and sh > best_one_sh:
            best_one, best_one_sh = c, sh
    best_one_ret = R.loc[idx_h2, best_one] if (best_one is not None and best_one_sh > 0) else pd.Series(0.0, index=idx_h2)

    # рівноважне усереднення всіх singles на H2
    mean_ret = R.loc[idx_h2].mean(axis=1)

    # oracle: ex-post найкраща на H2 (недосяжна межа)
    best_h2, best_h2_sh = None, -np.inf
    for c in cols:
        sh = sharpe(R.loc[idx_h2, c], 200)
        if not np.isnan(sh) and sh > best_h2_sh:
            best_h2, best_h2_sh = c, sh
    oracle_ret = R.loc[idx_h2, best_h2]
    # oracle-перемикання: у кожному режимі ex-post найкраща на H2
    oracle_switch = pd.Series(0.0, index=idx_h2)
    for key, _uk in REGIMES:
        sub = R.loc[lab_h2 == key]
        buk, bsh = None, -np.inf
        for c in cols:
            sh = sharpe(sub[c], 30)
            if not np.isnan(sh) and sh > bsh:
                buk, bsh = c, sh
        if buk is not None and bsh > 0:
            oracle_switch.loc[lab_h2 == key] = sub[buk]

    # чи збігається вибір H1 з найкращим на H2 (стабільність режимної карти)
    stable = []
    for key, _uk in REGIMES:
        sub_h2 = R.loc[lab_h2 == key]
        if len(sub_h2) < 30:
            continue
        best_h2_regime, bsh2 = None, -np.inf
        for c in cols:
            sh = sharpe(sub_h2[c], 30)
            if not np.isnan(sh) and sh > bsh2:
                best_h2_regime, bsh2 = c, sh
        stable.append(
            {
                "regime": key,
                "pick_h1": choice[key],
                "best_h2": best_h2_regime,
                "match": choice[key] == best_h2_regime,
            }
        )

    return {
        "symbol": symbol,
        "n_bars_h1": len(idx_h1),
        "n_bars_h2": len(idx_h2),
        "pick_flat": choice["range"] or "FLAT",
        "pick_bull": choice["trend_up"] or "FLAT",
        "pick_bear": choice["trend_down"] or "FLAT",
        "regime_map_matches": sum(1 for s in stable if s["match"]),
        "regime_map_cells": len(stable),
        "switch_sharpe": sharpe(switch_ret, 100),
        "switch_ret": float((1 + switch_ret).prod() - 1),
        "best_single_h1_sharpe": sharpe(best_one_ret, 100),
        "best_single_h1_ret": float((1 + best_one_ret).prod() - 1),
        "best_single_h1_name": best_one or "FLAT",
        "mean_all_sharpe": sharpe(mean_ret, 100),
        "mean_all_ret": float((1 + mean_ret).prod() - 1),
        "oracle_single_sharpe": sharpe(oracle_ret, 100),
        "oracle_single_name": best_h2 or "-",
        "oracle_switch_sharpe": sharpe(oracle_switch, 100),
        "flat_sharpe": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-glob", type=str, default="results/iter7_shard_*.csv")
    ap.add_argument("--report", type=str, default="")
    args = ap.parse_args()

    cells = load_cells(args.cells_glob)
    print(f"клітинок: {len(cells)}, ok={int((cells['status'] == 'ok').sum())}")
    rating = rate(cells)
    rating.to_csv(RATING_CSV, index=False)
    pd.set_option("display.width", 220)

    oos_by_symbol: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}
    for sym in sorted(cells["symbol"].unique()):
        try:
            oos_by_symbol[sym] = load_oos(sym)
        except SystemExit:
            continue

    reg = regime_matrix(oos_by_symbol)
    reg.to_csv(REGIME_CSV, index=False)

    singles = sorted({v for v in cells["variant"].unique() if v.startswith("single:")})
    metas = sorted({v for v in cells["variant"].unique() if v.startswith("meta:")})

    rows = []
    for sym, (mat, lab, _v) in oos_by_symbol.items():
        rows.append(switch_test(sym, mat, lab, singles))
    sw = pd.DataFrame(rows)
    sw.to_csv(SWITCH_CSV, index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "switch_sharpe",
                "best_single_h1_sharpe",
                "mean_all_sharpe",
                "oracle_single_sharpe",
                "oracle_switch_sharpe",
                "switch_ret",
                "best_single_h1_ret",
                "mean_all_ret",
            ],
            "mean": [sw[c].mean() for c in (
                "switch_sharpe",
                "best_single_h1_sharpe",
                "mean_all_sharpe",
                "oracle_single_sharpe",
                "oracle_switch_sharpe",
                "switch_ret",
                "best_single_h1_ret",
                "mean_all_ret",
            )],
            "median": [sw[c].median() for c in (
                "switch_sharpe",
                "best_single_h1_sharpe",
                "mean_all_sharpe",
                "oracle_single_sharpe",
                "oracle_switch_sharpe",
                "switch_ret",
                "best_single_h1_ret",
                "mean_all_ret",
            )],
        }
    )
    summary.to_csv(SWITCH_SUM_CSV, index=False)

    print("\n=== РЕЙТИНГ (агрегат по 10 символах) ===")
    print(rating.round(3).to_string(index=False))
    print("\n=== SHARPE per-regime (mean по символах) ===")
    piv = reg.pivot_table(index="variant", columns="regime", values="sharpe", aggfunc="mean")
    order = [r for r, _ in REGIMES]
    print(piv[[c for c in order if c in piv.columns]].round(3).to_string())
    print("\n=== ТЕСТ ПЕРЕМИКАННЯ H1→H2 ===")
    print(sw.round(3).to_string(index=False))
    print("\n", summary.round(3).to_string(index=False))
    print(f"\nCSV: {RATING_CSV}, {REGIME_CSV}, {SWITCH_CSV}, {SWITCH_SUM_CSV}")

    if args.report:
        from pathlib import Path as _P

        _P(args.report).write_text(
            "<!-- генерується experiments/iter7_regime_analysis.py -->\n"
            + rating.round(4).to_markdown(index=False)
            + "\n\n"
            + piv.round(4).to_markdown()
            + "\n\n"
            + sw.round(4).to_markdown(index=False),
            encoding="utf-8",
        )
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
