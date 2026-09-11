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
ROLL_CSV = Path("results/iter7_rolling_switch.csv")
DSR_CSV = Path("results/iter7_dsr_pbo.csv")

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
        lab_arr = lab.reindex(mat.index).fillna("range").to_numpy()
        for label in mat.columns:
            col = mat[label].to_numpy()
            for key, _uk in REGIMES:
                mask = lab_arr == key
                if int(mask.sum()) < 30:
                    continue
                rows.append({"symbol": symbol, "variant": label, "regime": key, "sharpe": sharpe(pd.Series(col[mask]), 30)})
    return pd.DataFrame(rows)


def _pick_best(sub: pd.DataFrame, cols: list[str], min_bars: int) -> tuple[str | None, float]:
    best_lab, best_sh = None, -np.inf
    for c in cols:
        s = sub[c]
        if int(s.notna().sum()) < min_bars:
            continue
        sh = sharpe(s, min_bars)
        if np.isnan(sh):
            continue
        if sh > best_sh:
            best_lab, best_sh = c, sh
    if best_lab is None or best_sh <= 0:
        return None, float(best_sh)
    return best_lab, float(best_sh)


def taxonomy_priors() -> dict[str, frozenset[str]]:
    from scalper_hft.strategies import REGISTRY

    return {name: frozenset(cls.preferred_regimes) for name, cls in REGISTRY.items()}


def _taxonomy_choice(cols: list[str], structure: str, priors: dict[str, frozenset[str]]) -> str | None:
    """Вибір стратегії за СТАТИЧНИМ таксономічним пріором (як робить best_prior).

    Лише стратегії з непорожнім preferred_regimes: у taxonomy порожній набір =
    «вага 1.0 у всіх режимах», тож універсальна стратегія завжди виграє тай-брейк
    і перемикання вироджується (це реальна властивість поточного best_prior).
    """
    from scalper_hft.strategies.taxonomy import regime_capital_weight

    best, best_w = None, -np.inf
    for c in cols:
        name = c.split(":", 1)[1]
        pref = priors.get(name, frozenset())
        if not pref:
            continue
        w = regime_capital_weight(pref, structure, "normal")
        if w > best_w:
            best, best_w = c, w
    return best


def rolling_switch(
    mat: pd.DataFrame,
    lab: pd.Series,
    pool: list[str],
    *,
    n_folds: int = 4,
) -> list[dict]:
    """Rolling-валідація селектора: карта режим→стратегія з фолду k → фолд k+1.

    Дає n_folds-1 незалежних OOS-оцінок селектора (замість одного спліту).
    Порівнює кілька селекторів на одному тестовому фолді:
      - empirical       — карта режим→найкраща стратегія на train-фолді (з пулу);
      - empirical_no_carry — те саме без funding_carry (перевірка залежності від
        одного експерта);
      - empirical_children — те саме, але пул = дефолтні діти supervisor
        (mean_reversion, supertrend, hmm_reversion);
      - taxonomy        — статичний пріор preferred_regimes (лише спеціалізовані);
      - bull_only       — найпростіший мета-фільтр: supertrend у trend_up, інакше flat;
      - best_single     — одна найкраща стратегія з train-фолду на весь тест-фолд;
      - mean_all        — рівноважний бленд усіх стратегій пулу.
    """
    cols = [c for c in pool if c in mat.columns]
    R = mat[cols].copy()
    R["__regime"] = lab.reindex(R.index).fillna("range").to_numpy()
    priors = taxonomy_priors()
    bounds = np.linspace(0, len(R), n_folds + 1).astype(int)
    children = [c for c in ("single:mean_reversion", "single:supertrend", "single:hmm_reversion") if c in cols]
    rows: list[dict] = []
    parts: list[pd.DataFrame] = []
    for k in range(n_folds - 1):
        tr = R.iloc[bounds[k] : bounds[k + 1]]
        te = R.iloc[bounds[k + 1] : bounds[k + 2]]
        if len(tr) < 500 or len(te) < 500:
            continue
        reg_tr = tr["__regime"].to_numpy()
        reg_te = te["__regime"].to_numpy()

        def build_map(cands: list[str]) -> dict[str, str | None]:
            m: dict[str, str | None] = {}
            for key, _uk in REGIMES:
                m[key], _ = _pick_best(tr.loc[reg_tr == key, cands], cands, 200)
            return m

        emp = build_map(cols)
        emp_nc = build_map([c for c in cols if c != "single:funding_carry"])
        emp_ch = build_map(children) if children else dict.fromkeys((r for r, _ in REGIMES))
        tax = {key: _taxonomy_choice(cols, key, priors) for key, _uk in REGIMES}

        def apply_map(m: dict[str, str | None]) -> pd.Series:
            # Лаг 1 бар: рішення на барі t (режим_t відомий на закритті t)
            # реалізується у дохідності бару t+1 — без lookahead.
            out = pd.Series(0.0, index=te.index)
            for key, _uk in REGIMES:
                c = m.get(key)
                if c is None:
                    continue
                pos = np.flatnonzero(reg_te == key)
                if len(pos):
                    out.iloc[pos] = te.loc[reg_te == key, c].to_numpy()
            return out.shift(1).fillna(0.0)

        emp_ret = apply_map(emp)
        emp_nc_ret = apply_map(emp_nc)
        emp_ch_ret = apply_map(emp_ch)
        tax_ret = apply_map(tax)

        # bull_only: momentum лише в бичому режимі, інакше flat
        bull_pick, _ = _pick_best(tr.loc[reg_tr == "trend_up", cols], cols, 200)
        bull_ret = pd.Series(0.0, index=te.index)
        if bull_pick is not None:
            pos = np.flatnonzero(reg_te == "trend_up")
            if len(pos):
                bull_ret.iloc[pos] = te.loc[reg_te == "trend_up", bull_pick].to_numpy()
        bull_ret = bull_ret.shift(1).fillna(0.0)

        best_one, _ = _pick_best(tr, cols, 500)
        bs_ret = te[best_one].set_axis(te.index).shift(1).fillna(0.0) if best_one is not None else pd.Series(0.0, index=te.index)
        mean_ret = te[cols].mean(axis=1).set_axis(te.index)

        rows.append(
            {
                "fold": k + 1,
                "n_te_bars": len(te),
                "empirical_sharpe": sharpe(emp_ret, 300),
                "empirical_ret": float((1 + emp_ret).prod() - 1),
                "empirical_no_carry_sharpe": sharpe(emp_nc_ret, 300),
                "empirical_children_sharpe": sharpe(emp_ch_ret, 300),
                "bull_only_sharpe": sharpe(bull_ret, 300),
                "taxonomy_sharpe": sharpe(tax_ret, 300),
                "taxonomy_ret": float((1 + tax_ret).prod() - 1),
                "best_single_sharpe": sharpe(bs_ret, 300),
                "best_single_name": best_one or "FLAT",
                "mean_all_sharpe": sharpe(mean_ret, 300),
                "emp_map": f"flat={emp['range']}|bull={emp['trend_up']}|bear={emp['trend_down']}",
                "tax_map": f"flat={tax['range']}|bull={tax['trend_up']}|bear={tax['trend_down']}",
            }
        )
        parts.append(
            pd.DataFrame(
                {
                    "empirical": emp_ret,
                    "best_single": bs_ret,
                    "mean_all": mean_ret,
                    "taxonomy": tax_ret,
                    "bull_only": bull_ret,
                }
            )
        )
    series = pd.concat(parts) if parts else pd.DataFrame()
    return rows, series


def dsr_table(
    oos_by_symbol: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]],
    singles: list[str],
    *,
    n_folds: int = 4,
) -> pd.DataFrame:
    """DSR + PBO для трьох підходів: селектор-за-режимом vs best-single vs mean-all.

    n_trials для селектора = 3 режими × N стратегій (стільки комбінацій реально
    перебрано); для best-single — N стратегій. DSR рахується на конкатенованих
    дохідностях rolling test-фолдів (уже OOS, без IS-забруднення).
    """
    from scalper_hft.validation.cscv import pbo_cscv
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio

    rows: list[dict] = []
    for sym, (mat, lab, _v) in oos_by_symbol.items():
        _rows, series = rolling_switch(mat, lab, singles, n_folds=n_folds)
        if series.empty:
            continue
        cols = [c for c in singles if c in mat.columns]
        n_trials = 3 * len(cols)
        rec: dict = {"symbol": sym, "n_test_bars": len(series)}
        for name, col in (("selector", "empirical"), ("best_single", "best_single"), ("mean_all", "mean_all")):
            r = series[col].dropna()
            rec[f"dsr_{name}"] = float(deflated_sharpe_ratio(r, n_trials=n_trials if name != "mean_all" else 1))
            rec[f"sharpe_{name}"] = sharpe(r, 300)
        # PBO: наскільки «вибір найкращого варіанта» схильний до перенавчання
        sub = mat[cols].reindex(series.index).fillna(0.0)
        arr = np.vstack([sub.to_numpy().T, series["empirical"].to_numpy()[None, :]])
        try:
            rec["pbo_cscv"] = float(pbo_cscv(arr, n_blocks=8, purge_bars=0).pbo)
        except Exception as exc:  # noqa: BLE001
            rec["pbo_cscv"] = float("nan")
            rec["pbo_error"] = str(exc)[:120]
        rows.append(rec)
    return pd.DataFrame(rows)


def switch_test(
    symbol: str,
    mat: pd.DataFrame,
    lab: pd.Series,
    pool: list[str],
) -> dict:
    """Селектор режим→стратегія, навчений на H1, застосований до H2."""
    cols = [c for c in pool if c in mat.columns]
    R = mat[cols].copy()
    R["__regime"] = lab.reindex(R.index).fillna("range").to_numpy()
    n = len(R)
    half = n // 2
    h1, h2 = R.iloc[:half], R.iloc[half:]
    idx_h2 = h2.index
    reg_h1 = h1["__regime"].to_numpy()
    reg_h2 = h2["__regime"].to_numpy()

    # ── селектор: найкраща стратегія per-regime на H1 ──────────────────────
    choice: dict[str, str | None] = {}
    for key, _uk in REGIMES:
        sub = h1.loc[reg_h1 == key, cols]
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
        mask = reg_h2 == key
        switch_ret.iloc[np.flatnonzero(mask)] = h2.loc[mask, c].to_numpy()

    # базова лінія: одна стратегія на весь H2, обрана на H1 (без перемикання)
    best_one, best_one_sh = None, -np.inf
    for c in cols:
        sh = sharpe(h1[c], 200)
        if not np.isnan(sh) and sh > best_one_sh:
            best_one, best_one_sh = c, sh
    if best_one is not None and best_one_sh > 0:
        best_one_ret = h2[best_one].set_axis(idx_h2)
    else:
        best_one_ret = pd.Series(0.0, index=idx_h2)

    # рівноважне усереднення всіх singles на H2
    mean_ret = h2[cols].mean(axis=1).set_axis(idx_h2)

    # oracle: ex-post найкраща на H2 (недосяжна межа)
    best_h2, best_h2_sh = None, -np.inf
    for c in cols:
        sh = sharpe(h2[c], 200)
        if not np.isnan(sh) and sh > best_h2_sh:
            best_h2, best_h2_sh = c, sh
    oracle_ret = h2[best_h2].set_axis(idx_h2)
    # oracle-перемикання: у кожному режимі ex-post найкраща на H2
    oracle_switch = pd.Series(0.0, index=idx_h2)
    for key, _uk in REGIMES:
        mask = reg_h2 == key
        sub = h2.loc[mask, cols]
        buk, bsh = None, -np.inf
        for c in cols:
            sh = sharpe(sub[c], 30)
            if not np.isnan(sh) and sh > bsh:
                buk, bsh = c, sh
        if buk is not None and bsh > 0:
            oracle_switch.iloc[np.flatnonzero(mask)] = sub[buk].to_numpy()

    # чи збігається вибір H1 з найкращим на H2 (стабільність режимної карти)
    stable = []
    for key, _uk in REGIMES:
        sub_h2 = h2.loc[reg_h2 == key, cols]
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
        "n_bars_h1": len(h1),
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

    rows = []
    for sym, (mat, lab, _v) in oos_by_symbol.items():
        rows.append(switch_test(sym, mat, lab, singles))
    sw = pd.DataFrame(rows)
    sw.to_csv(SWITCH_CSV, index=False)

    roll_rows: list[dict] = []
    for sym, (mat, lab, _v) in oos_by_symbol.items():
        rows_sym, _series = rolling_switch(mat, lab, singles, n_folds=4)
        for r in rows_sym:
            roll_rows.append({"symbol": sym, **r})
    roll = pd.DataFrame(roll_rows)
    roll.to_csv(ROLL_CSV, index=False)

    dsr = dsr_table(oos_by_symbol, singles)
    dsr.to_csv(DSR_CSV, index=False)

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
    print("\n=== ROLLING-ВАЛІДАЦІЯ СЕЛЕКТОРА (фолд k → k+1), агрегат ===")
    if not roll.empty:
        roll_agg = (
            roll.groupby("fold")
            .agg(
                n=("symbol", "count"),
                empirical=("empirical_sharpe", "mean"),
                emp_no_carry=("empirical_no_carry_sharpe", "mean"),
                emp_children=("empirical_children_sharpe", "mean"),
                bull_only=("bull_only_sharpe", "mean"),
                taxonomy=("taxonomy_sharpe", "mean"),
                best_single=("best_single_sharpe", "mean"),
                mean_all=("mean_all_sharpe", "mean"),
            )
            .round(3)
        )
        print(roll_agg.to_string())
        print(
            "\nсереднє по всіх фолдах: "
            f"empirical={roll['empirical_sharpe'].mean():+.3f} "
            f"emp_no_carry={roll['empirical_no_carry_sharpe'].mean():+.3f} "
            f"emp_children={roll['empirical_children_sharpe'].mean():+.3f} "
            f"bull_only={roll['bull_only_sharpe'].mean():+.3f} "
            f"taxonomy={roll['taxonomy_sharpe'].mean():+.3f} "
            f"best_single={roll['best_single_sharpe'].mean():+.3f} "
            f"mean_all={roll['mean_all_sharpe'].mean():+.3f}"
        )
        print(
            "частка фолдів з empirical > best_single: "
            f"{float((roll['empirical_sharpe'] > roll['best_single_sharpe']).mean()):.0%}; "
            "empirical > 0: "
            f"{float((roll['empirical_sharpe'] > 0).mean()):.0%}"
        )
        print("\nтипові карти режимів (empirical):")
        print(roll["emp_map"].value_counts().head(6).to_string())
        print("\nтипові карти режимів (taxonomy):")
        print(roll["tax_map"].value_counts().head(6).to_string())
    print("\n=== DSR / PBO (rolling test-фолди, per symbol) ===")
    if not dsr.empty:
        print(dsr.round(3).to_string(index=False))
        print(
            "\nсереднє: "
            f"DSR selector={dsr['dsr_selector'].mean():.3f} "
            f"(>0.95 у {int((dsr['dsr_selector'] > 0.95).sum())}/{len(dsr)} символів), "
            f"DSR best_single={dsr['dsr_best_single'].mean():.3f}, "
            f"DSR mean_all={dsr['dsr_mean_all'].mean():.3f}, "
            f"PBO={dsr['pbo_cscv'].mean():.3f}"
        )
    print(f"\nCSV: {RATING_CSV}, {REGIME_CSV}, {SWITCH_CSV}, {SWITCH_SUM_CSV}, {ROLL_CSV}, {DSR_CSV}")

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
