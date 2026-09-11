"""Iteration 7 — рейтинг стратегій і regime-адаптивних мета-стратегій (1h, LIVE-дані).

Питання: які стратегії мають більше шансів, і чи дає перевагу мета-сутність, що
перемикає стратегію за типом ринку (бичий trend_up / ведмежий trend_down / флет range).

Дизайн (без lookahead, без підгонки):
    - дані: LIVE-кеш (`data_exchange=binanceusdm`), 1h деривація з 1m, days=1095;
    - walk-forward train=2500/test=500, ФІКСОВАНІ дефолтні параметри (без IS-оптимізації);
    - maker-виконання (висновок проєкту: taker на суб-годинних барах не виживає);
    - collect_oos_returns=True → конкатеновані OOS бар-дохідності для
      per-regime атрибуції (режимні мітки каузальні: RegimeDetector);
    - жодна клітинка не «вибирається» за результатом: рейтинг рахується на
      всіх символах, метрики агрегуються (mean/median), а не max.

Варіанти:
    singles  — одиночні альфи (momentum / mean-reversion / flow / carry);
    metas    — ensemble (mean|vote|hedge|regime) і regime_supervisor
               (regime_soft | best_prior | contextual_hedge | exp3 + гейти/гістерезис/HTF).

Запуск (з кореня репо):
    .venv/bin/python experiments/iter7_regime_rating.py --workers 5

Результати:
    results/iter7_regime_rating.csv        — метрики клітинок (symbol × variant)
    results/iter7_oos/<sym>__<variant>.parquet — OOS бар-дохідності + режимні мітки
    Аналіз: experiments/iter7_regime_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS_DEFAULT = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,LINKUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,UNIUSDT"
INTERVAL = "1h"
BASE = "1m"
PERIODS_PER_YEAR = 8760.0  # 1h барах
OOS_DIR = Path("results/iter7_oos")
OUT_CSV = Path("results/iter7_regime_rating.csv")

# Діти мета-стратегій: тренд (supertrend) + флет (stoch_rsi, mean_reversion).
# cross_momentum виключено — вимагає multi_symbol (fail-fast strict_data).
CHILDREN_SPLIT = "supertrend,stoch_rsi,mean_reversion"


def build_variants() -> list[tuple[str, dict]]:
    """Список (label, spec). spec: name/group/adaptive + kwargs."""
    v: list[tuple[str, dict]] = [
        # ── singles ────────────────────────────────────────────────────────
        ("single:supertrend", {"name": "supertrend", "group": "single_momentum", "adaptive": False}),
        ("single:stoch_rsi", {"name": "stoch_rsi", "group": "single_meanrev", "adaptive": False}),
        ("single:mean_reversion", {"name": "mean_reversion", "group": "single_meanrev", "adaptive": False}),
        ("single:hmm_reversion", {"name": "hmm_reversion", "group": "single_meanrev", "adaptive": False}),
        ("single:smc_fvg", {"name": "smc_fvg", "group": "single_flow", "adaptive": False}),
        ("single:cvd_momentum", {"name": "cvd_momentum", "group": "single_flow", "adaptive": False}),
        ("single:ob_imbalance", {"name": "ob_imbalance", "group": "single_flow", "adaptive": False}),
        ("single:funding_carry", {"name": "funding_carry", "group": "single_carry", "adaptive": False}),
        ("single:basis_reversion", {"name": "basis_reversion", "group": "single_carry", "adaptive": False}),
        # ── meta: ensemble (без режиму) ────────────────────────────────────
        (
            "meta:ens_mean",
            {"name": "ensemble", "group": "meta_static", "adaptive": False, "children": CHILDREN_SPLIT, "mode": "mean"},
        ),
        (
            "meta:ens_vote",
            {"name": "ensemble", "group": "meta_static", "adaptive": False, "children": CHILDREN_SPLIT, "mode": "vote"},
        ),
        (
            "meta:ens_hedge",
            {"name": "ensemble", "group": "meta_online", "adaptive": True, "children": CHILDREN_SPLIT, "mode": "hedge"},
        ),
        (
            "meta:ens_regime",
            {
                "name": "ensemble",
                "group": "meta_regime",
                "adaptive": True,
                "children": CHILDREN_SPLIT,
                "mode": "regime",
            },
        ),
        # ── meta: regime_supervisor (перемикання за типом ринку) ────────────
        (
            "meta:sup_regime_soft",
            {
                "name": "regime_supervisor",
                "group": "meta_regime",
                "adaptive": True,
                "children": CHILDREN_SPLIT,
                "kwargs": {"blend_mode": "regime_soft"},
            },
        ),
        (
            "meta:sup_best_prior",
            {
                "name": "regime_supervisor",
                "group": "meta_regime",
                "adaptive": True,
                "children": CHILDREN_SPLIT,
                "kwargs": {"blend_mode": "best_prior"},
            },
        ),
        (
            "meta:sup_ctx_hedge",
            {
                "name": "regime_supervisor",
                "group": "meta_regime",
                "adaptive": True,
                "children": CHILDREN_SPLIT,
                "kwargs": {"blend_mode": "contextual_hedge"},
            },
        ),
        (
            "meta:sup_exp3",
            {
                "name": "regime_supervisor",
                "group": "meta_online",
                "adaptive": True,
                "children": CHILDREN_SPLIT,
                "kwargs": {"blend_mode": "exp3"},
            },
        ),
        (
            "meta:sup_best_prior_dwell3",
            {
                "name": "regime_supervisor",
                "group": "meta_regime",
                "adaptive": True,
                "children": CHILDREN_SPLIT,
                "kwargs": {"blend_mode": "best_prior", "min_dwell_bars": 3},
            },
        ),
        (
            "meta:sup_best_prior_lazy",
            {
                "name": "regime_supervisor",
                "group": "meta_regime",
                "adaptive": True,
                "children": CHILDREN_SPLIT,
                "kwargs": {"blend_mode": "best_prior", "lazy_gating": True},
            },
        ),
        (
            "meta:sup_regime_soft_htf1d",
            {
                "name": "regime_supervisor",
                "group": "meta_regime_htf",
                "adaptive": True,
                "children": CHILDREN_SPLIT,
                "kwargs": {"blend_mode": "regime_soft", "htf_tf": "1d"},
            },
        ),
    ]
    return v


def build_strategy(spec: dict):
    from scalper_hft.strategies import get_strategy

    if spec["name"] == "regime_supervisor":
        kwargs = dict(spec.get("kwargs", {}))
        return get_strategy(
            "regime_supervisor",
            strategies=spec["children"],
            n_hmm_states=3,
            hmm_fit_bars=2000,
            **kwargs,
        )
    if spec["name"] == "ensemble":
        return get_strategy("ensemble", strategies=spec["children"], mode=spec["mode"])
    return get_strategy(spec["name"], **spec.get("kwargs", {}))


def _sharpe(returns: pd.Series, min_bars: int = 30) -> float:
    r = returns.dropna()
    if len(r) < min_bars:
        return float("nan")
    sd = float(r.std(ddof=0))
    if sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(PERIODS_PER_YEAR))


# Кеш даних на рівень процесу (серіальний режим: один процес = один набір символів).
# Без нього кожна клітинка перечитує 1m-parquet і ресемплить 1.5M барів.
_DATA_CACHE: dict[tuple[str, str, int], tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]] = {}
_REGIME_CACHE: dict[str, pd.DataFrame] = {}


def _load_symbol(symbol: str, days: int, *, needs_trades: bool, needs_funding: bool):
    key = (symbol, INTERVAL, days)
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.store import get_store

    cached = _DATA_CACHE.get(key)
    if cached is None:
        klines = ensure_klines(symbol, INTERVAL, days, base_interval=BASE, derive=True, readonly=True)
        _DATA_CACHE[key] = (klines, None, None)
        cached = _DATA_CACHE[key]
    klines, trades, funding = cached
    if needs_trades and trades is None:
        trades = get_store().load_trades(symbol)
        _DATA_CACHE[key] = (klines, trades, funding)
    if needs_funding and funding is None:
        funding = get_store().load_funding(symbol)
        _DATA_CACHE[key] = (klines, trades, funding)
    return klines, trades, funding


def run_cell(
    symbol: str,
    label: str,
    spec: dict,
    days: int,
    train: int,
    test: int,
    maker: bool,
    oos_dir: str,
) -> dict:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.features.regime_detector import RegimeDetector
    from scalper_hft.validation.walk_forward import run_walk_forward

    row: dict = {
        "symbol": symbol,
        "variant": label,
        "group": spec["group"],
        "adaptive": spec["adaptive"],
        "strategy": spec["name"],
        "status": "ok",
        "error": "",
    }
    t0 = time.time()
    try:
        settings = get_settings()
        strat = build_strategy(spec)
        df, trades, funding = _load_symbol(
            symbol,
            days,
            needs_trades=strat.needs_trades,
            needs_funding=strat.needs_funding,
        )
        cost = CostModel(
            maker_fee=settings.maker_fee,
            taker_fee=settings.taker_fee,
            slippage_frac=settings.slippage_frac,
        )
        res = run_walk_forward(
            df,
            strat,
            train,
            test,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
            interval=INTERVAL,
            is_maker=maker,
            collect_oos_returns=True,
        )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = str(exc)[:300]
        row["seconds"] = round(time.time() - t0, 1)
        return row

    oos = res.oos_returns
    row.update(
        {
            "n_bars": int(len(df)),
            "n_windows": len(res.windows),
            "n_trades_oos": int(sum(w.n_trades for w in res.windows)),
            "avg_is_sharpe": round(float(res.avg_is_sharpe), 4),
            "avg_oos_sharpe": round(float(res.avg_oos_sharpe), 4),
            "oos_positive_frac": round(float(res.positive_windows_frac), 4),
            "seconds": round(time.time() - t0, 1),
        }
    )

    if oos is None or oos.empty:
        row["status"] = "no_oos_returns"
        return row

    regime = _REGIME_CACHE.get(symbol)
    if regime is None:
        regime = RegimeDetector(n_hmm_states=3, hmm_fit_bars=2000).detect(df["close"])
        _REGIME_CACHE[symbol] = regime
    reg = regime[["structure", "vol", "label"]].reindex(oos.index)
    frame = pd.DataFrame({"ret": oos.astype(float), "structure": reg["structure"].fillna("range")})
    frame = frame[~frame.index.duplicated(keep="first")]
    out = Path(oos_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / f"{symbol}__{label.replace(':', '_')}.parquet")

    equity = (1.0 + frame["ret"]).cumprod()
    row["pooled_oos_sharpe"] = round(_sharpe(frame["ret"]), 4)
    row["oos_total_return"] = round(float(equity.iloc[-1] - 1.0), 4)
    row["oos_max_dd"] = round(float((equity / equity.cummax() - 1.0).min()), 4)
    for tag, lab in (("range", "range"), ("up", "trend_up"), ("down", "trend_down")):
        sub = frame.loc[frame["structure"] == lab, "ret"]
        row[f"sharpe_{tag}"] = round(_sharpe(sub, min_bars=50), 4)
        row[f"n_bars_{tag}"] = int(len(sub))

    # Дегенерація (AGENTS.md; experiments/README.md: «Мінімум 30 угод за 3y на
    # клітинку — інакше degenerate»). Без цього клітинки з 0 угод писались як
    # `status="ok"` і Sharpe рівно 0.0, потрапляли в рейтинг як «нульовий edge»
    # і в пул regime-селектора. Аудит 2026-09-11: так було 57 із 200 клітинок.
    from scalper_hft.validation.cell_audit import min_trades_for

    n_tr = int(row["n_trades_oos"])
    min_tr = min_trades_for(INTERVAL)
    if n_tr <= 0:
        row["status"] = "degenerate"
        row["error"] = "0 угод — стратегія не торгувала"
    elif min_tr and n_tr < min_tr:
        row["status"] = "degenerate"
        row["error"] = f"угод {n_tr} < {min_tr} — статистично шум"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1095)
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--symbols", type=str, default=SYMBOLS_DEFAULT)
    ap.add_argument("--variants", type=str, default="", help="фільтр label через кому (порожньо = всі)")
    ap.add_argument("--out-csv", type=str, default=str(OUT_CSV), help="куди писати CSV клітинок")
    ap.add_argument("--oos-dir", type=str, default=str(OOS_DIR), help="куди писати OOS бар-дохідності")
    args = ap.parse_args()
    out_csv = Path(args.out_csv)

    symbols = [s for s in args.symbols.split(",") if s]
    variants = build_variants()
    if args.variants:
        keep = {v.strip() for v in args.variants.split(",") if v.strip()}
        variants = [(label, spec) for label, spec in variants if label in keep]

    cells = [
        (sym, label, spec, args.days, args.train, args.test, True, args.oos_dir)
        for sym in symbols
        for label, spec in variants
    ]
    print(f"клітинок: {len(cells)} ({len(symbols)} символів × {len(variants)} варіантів)", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    if args.workers <= 1:
        # Серіальний режим (пісочниця без /dev/shm → ProcessPool недоступний).
        for i, c in enumerate(cells, 1):
            rows.append(run_cell(*c))
            if i % 10 == 0 or i == len(cells):
                print(f"  {i}/{len(cells)} готово, {time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_cell, *c) for c in cells]
            for i, f in enumerate(futures, 1):
                rows.append(f.result())
                if i % 10 == 0 or i == len(futures):
                    print(f"  {i}/{len(futures)} готово, {time.time() - t0:.0f}s", flush=True)

    rows.sort(key=lambda r: (r["symbol"], r["variant"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    print(f"\nГотово за {time.time() - t0:.0f}s. ok={len(ok)}, error={len(df) - len(ok)}")
    if not ok.empty:
        agg = (
            ok.groupby("variant")
            .agg(
                n_symbols=("symbol", "count"),
                mean_oos_sharpe=("avg_oos_sharpe", "mean"),
                mean_pooled=("pooled_oos_sharpe", "mean"),
                pos_frac=("oos_positive_frac", "mean"),
                trades=("n_trades_oos", "sum"),
            )
            .sort_values("mean_oos_sharpe", ascending=False)
        )
        print(agg.round(3).to_string())
    print(f"\nCSV: {out_csv}\nOOS: {args.oos_dir}")


if __name__ == "__main__":
    main()
