#!/usr/bin/env python
"""Повний overfitting-аудит ПАРИ (pairs_arb) — те, що `cli overfit` не вміє.

Причина існування: `cli overfit --strategy pairs_arb` падає з
`MissingDataError: pairs_arb: заявлено ['multi_symbol','ohlcv']` — `audit_cell`
вантажить ОДИН символ, а pairs_arb потребує дві ноги (leg1+leg2). Через це
недосяжним був і `_maybe_record_pair_verdict` (він викликається ПІСЛЯ успішного
cell-аудиту, який для pairs завжди error) → pair-вердикт для live-гейта нічим
не оновити.

Що рахує цей скрипт (критерії validation/pairs_gate.py + AFML):
    1. Повний бектест пари (maker/taker, обидва funding-потоки) — SRh/Sharpe/DD/PF.
    2. Walk-forward з ПО-ВІКОННИМИ метриками (IS/OOS Sharpe, return, угоди) —
       частка позитивних OOS-вікон = головний критерій pairs-гейта (>= 55%).
    3. Deflated Sharpe на КОНКАТЕНОВАНИХ OOS-дохідностях (не на full-sample —
       full-sample забруднений IS і системно завищує DSR).
    4. CSCV/PBO (Bailey–López de Prado) по варіантах параметрів на парі.
    5. Sensitivity (smoothness) головного параметра на OOS-регіоні.
    6. Stress-сценарії + benchmark buy&hold обох ніг.
    7. Pair-вердикт через evaluate_pair_wf_gate(+запис у verdict_store).

Запуск:
    uv run python scripts/pair_audit.py --leg1 LINKUSDT --leg2 BTCUSDT \
        --interval 1h --days 1095 --maker
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("pair_audit")

# Той самий дефолт вікон, що у sweep/аудиті для 1h (WF_TRAIN_TEST у cell_audit).
DEFAULT_TRAIN_BARS = 1500
DEFAULT_TEST_BARS = 500


def _leg_slice(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    return df.reindex(index).dropna()


def _fund_slice(funding: pd.DataFrame | None, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame | None:
    if funding is None or funding.empty:
        return None
    return funding[(funding.index >= t0) & (funding.index <= t1)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfitting-аудит пари pairs_arb")
    parser.add_argument("--leg1", default="LINKUSDT")
    parser.add_argument("--leg2", default="BTCUSDT")
    parser.add_argument("--strategy", default="pairs_arb")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN_BARS)
    parser.add_argument("--test", type=int, default=DEFAULT_TEST_BARS)
    parser.add_argument("--position-pct", type=float, default=0.3)
    parser.add_argument("--maker", action="store_true", default=True)
    parser.add_argument("--taker", dest="maker", action="store_false")
    parser.add_argument("--variants", type=int, default=20, help="варіантів для CSCV/PBO")
    parser.add_argument("--blocks", type=int, default=10, help="блоків CSCV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-record", action="store_true", help="не писати вердикт у verdict_store")
    parser.add_argument("--validated", action="store_true", help="взяти параметри пари з VALIDATED_PAIRS")
    parser.add_argument("-p", "--param", action="append", default=[], help="Параметр стратегії: key=value")
    parser.add_argument("--out", default=None, help="markdown-звіт (за замовч. docs/reports/)")
    args = parser.parse_args()

    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs import _align, _extract_leg_df, run_pairs_backtest
    from scalper_hft.cli import _load_klines
    from scalper_hft.config import get_settings
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.benchmark import buy_and_hold_sharpe
    from scalper_hft.validation.cscv import pbo_cscv
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio
    from scalper_hft.validation.pairs_gate import evaluate_pair_wf_gate
    from scalper_hft.validation.stress import stress_report
    from scalper_hft.validation.trial_ledger import default_path as ledger_default_path
    from scalper_hft.validation.trial_ledger import effective_n_trials

    settings = get_settings()
    t_start = time.monotonic()
    leg1, leg2 = args.leg1, args.leg2

    print("═" * 78)
    print(f"PAIR AUDIT: {args.strategy} {leg1}/{leg2} {args.interval}, {args.days} днів")
    print(
        f"Дані: DATA_EXCHANGE={getattr(settings, 'data_exchange', '?')} (LIVE) | торговий EXCHANGE={settings.exchange}"
    )
    print(
        f"Витрати: maker={settings.maker_fee:.4%} taker={settings.taker_fee:.4%} "
        f"slippage={settings.slippage_bps}bps | execution={'MAKER' if args.maker else 'TAKER'}"
    )
    print("═" * 78)

    # ── 1. Дані ───────────────────────────────────────────────────────────────
    df1 = _load_klines(leg1, args.interval, args.days, derive=True)
    df2 = _load_klines(leg2, args.interval, args.days, derive=True)
    f1 = download_funding(leg1, args.days)
    f2 = download_funding(leg2, args.days)
    strategy = get_strategy(args.strategy)
    params = dict(strategy.params)
    if args.validated:
        from scalper_hft.live.pairs_runner import VALIDATED_PAIRS

        match = next((c for c in VALIDATED_PAIRS if c["leg1"] == leg1 and c["leg2"] == leg2), None)
        if match is None:
            print(f"⚠ {leg1}/{leg2} немає у VALIDATED_PAIRS — лишаю дефолтні параметри")
        else:
            params.update({k: v for k, v in match.items() if k not in {"leg1", "leg2"}})
    for kv in args.param:
        key, _, raw = kv.partition("=")
        if not _:
            print(f"⚠ --param {kv}: очікую key=value")
            continue
        val: object = raw
        if raw.lower() in {"true", "false"}:
            val = raw.lower() == "true"
        else:
            try:
                val = int(raw) if raw.isdigit() else float(raw)
            except ValueError:
                val = raw
        params[key] = val
    strategy = type(strategy)(**params)
    print(
        f"\n[0] ДАНІ: {leg1} {len(df1)} барів ({df1.index[0]} … {df1.index[-1]}) | "
        f"{leg2} {len(df2)} барів ({df2.index[0]} … {df2.index[-1]})"
    )
    common = _align(df1, df2)
    print(f"    спільних барів: {len(common)} ({common.index[0]} … {common.index[-1]})")
    print(f"    параметри стратегії: {params}")

    # ── 2. Повний бектест ─────────────────────────────────────────────────────
    cost = CostModel.from_settings(settings, df=df1)
    bt = run_pairs_backtest(
        df1, df2, strategy, f1, f2, position_pct=args.position_pct, cost=cost, maker_execution=args.maker
    )
    m = bt.metrics
    bt_max_dd = abs(float(m.max_drawdown))
    print("\n[1] ПОВНИЙ БЕКТЕСТ (in-sample, ІНФО — забруднений підгонкою)")
    print(
        f"    return {m.total_return:+.2%} | Sharpe(річн) {m.sharpe:.2f} | SRh {m.sharpe_hourly:+.4f} | "
        f"maxDD {m.max_drawdown:.2%} | угод {int(m.n_trades)} | PF {m.profit_factor:.2f} | win {m.win_rate:.0%}"
    )
    print(f"    funding PnL: {bt.funding_pnl:+.3f}% | експозиція {m.exposure:.0%}")

    # ── 3. Walk-forward з по-віконними метриками ──────────────────────────────
    n_common = len(common)
    if n_common < args.train + args.test:
        print(f"\n[2] WALK-FORWARD: ЗАМАЛО ДАНИХ ({n_common} < {args.train + args.test})")
        return 1
    windows: list[dict] = []
    oos_returns: list[pd.Series] = []
    start = 0
    while start + args.train + args.test <= n_common:
        tr = common.iloc[start : start + args.train]
        te = common.iloc[start + args.train : start + args.train + args.test]
        l1_tr, l2_tr = _extract_leg_df(tr, 1), _extract_leg_df(tr, 2)
        l1_te, l2_te = _extract_leg_df(te, 1), _extract_leg_df(te, 2)
        try:
            r_is = run_pairs_backtest(
                l1_tr,
                l2_tr,
                strategy,
                _fund_slice(f1, tr.index[0], tr.index[-1]),
                _fund_slice(f2, tr.index[0], tr.index[-1]),
                position_pct=args.position_pct,
                cost=cost,
                maker_execution=args.maker,
            )
            r_oos = run_pairs_backtest(
                l1_te,
                l2_te,
                strategy,
                _fund_slice(f1, te.index[0], te.index[-1]),
                _fund_slice(f2, te.index[0], te.index[-1]),
                position_pct=args.position_pct,
                cost=cost,
                maker_execution=args.maker,
            )
            windows.append(
                {
                    "idx": len(windows),
                    "test_start": str(te.index[0]),
                    "test_end": str(te.index[-1]),
                    "is_sharpe": float(r_is.metrics.sharpe_hourly),
                    "oos_sharpe": float(r_oos.metrics.sharpe_hourly),
                    "oos_return": float(r_oos.metrics.total_return),
                    "oos_maxdd": float(r_oos.metrics.max_drawdown),
                    "n_trades": int(r_oos.metrics.n_trades),
                }
            )
            oos_returns.append(r_oos.equity.pct_change().dropna())
        except Exception as exc:  # noqa: BLE001
            logger.warning("вікно %d: %s", len(windows), exc)
            windows.append(
                {
                    "idx": len(windows),
                    "test_start": str(te.index[0]),
                    "test_end": str(te.index[-1]),
                    "is_sharpe": 0.0,
                    "oos_sharpe": 0.0,
                    "oos_return": 0.0,
                    "oos_maxdd": 0.0,
                    "n_trades": 0,
                }
            )
        start += args.test

    oos_sh = np.array([w["oos_sharpe"] for w in windows], dtype=float)
    is_sh = np.array([w["is_sharpe"] for w in windows], dtype=float)
    pos_frac = float((oos_sh > 0).mean()) if len(oos_sh) else 0.0
    n_trades_oos = int(sum(w["n_trades"] for w in windows))
    print(f"\n[2] WALK-FORWARD (train={args.train} / test={args.test} барів)")
    print(
        f"    вікон: {len(windows)} | avg IS SRh {is_sh.mean():+.4f} | avg OOS SRh {oos_sh.mean():+.4f} | "
        f"позитивних OOS: {pos_frac:.0%} | OOS угод: {n_trades_oos}"
    )
    print(
        f"    OOS SRh медіана {np.median(oos_sh):+.4f} | мін {oos_sh.min():+.4f} | макс {oos_sh.max():+.4f} | "
        f"частка OOS-вікон зі збитком: {(oos_sh < 0).mean():.0%}"
    )

    # ── 4. Deflated Sharpe на OOS ─────────────────────────────────────────────
    oos_cat = pd.concat(oos_returns) if oos_returns else pd.Series(dtype=float)
    ledger_path = getattr(settings, "trial_ledger_path", None) or ledger_default_path()
    combos_audit = 1 + args.variants + 6  # baseline + CSCV-варіанти + sensitivity-сітка
    n_trials = max(
        combos_audit,
        int(effective_n_trials(Path(ledger_path), param_combinations=combos_audit, strategy=args.strategy)),
    )
    dsr = deflated_sharpe_ratio(oos_cat.values, n_trials=n_trials) if len(oos_cat) >= 2 else 0.0
    print("\n[3] DEFLATED SHARPE (конкатеновані OOS-дохідності)")
    print(f"    DSR {dsr:.4f} | n_trials {n_trials} | OOS-барів {len(oos_cat)} | поріг DSR > 0.95")

    # ── 5. CSCV / PBO ─────────────────────────────────────────────────────────
    print(f"\n[4] CSCV PBO ({args.variants} варіантів параметрів × {args.blocks} блоків)")
    rng = np.random.default_rng(args.seed)
    space = strategy.param_space
    rows: list[np.ndarray] = []
    used: list[dict] = []
    for _ in range(args.variants):
        p = dict(params)
        for pname, (lo, hi, step) in space.items():
            if float(step) == int(step) and float(lo) == int(lo) and float(hi) == int(hi):
                p[pname] = int(rng.integers(int(lo), int(hi) + 1))
            else:
                p[pname] = float(rng.uniform(lo, hi))
        try:
            strat_v = type(strategy)(**p)
            res_v = run_pairs_backtest(
                df1, df2, strat_v, f1, f2, position_pct=args.position_pct, cost=cost, maker_execution=args.maker
            )
            eq = res_v.equity.pct_change().fillna(0.0)
            rows.append(eq.to_numpy(dtype=float))
            used.append({k: p[k] for k in space})
        except Exception as exc:  # noqa: BLE001
            logger.warning("варіант %s: %s", p, exc)
    pbo = float("nan")
    pbo_res = None
    if len(rows) >= 4:
        variant_mat = np.vstack(rows)
        n_bars = variant_mat.shape[1]
        auto_pe = max(1, n_bars // 100)
        pbo_res = pbo_cscv(variant_mat, n_blocks=args.blocks, threshold=0.0, purge_bars=auto_pe, embargo_bars=auto_pe)
        pbo = float(pbo_res.pbo)
        print(
            f"    варіантів пораховано: {len(rows)} | PBO {pbo:.3f} "
            f"({'✅ прийнятний (<0.50)' if pbo < 0.50 else '⚠ перенавчання ймовірне'})"
        )
        print(
            f"    OOS Sharpe IS-кращих: медіана {np.median(pbo_res.is_best_oos_sharpes):+.4f}, "
            f"частка < 0: {(pbo_res.is_best_oos_sharpes < 0).mean():.0%} | purge/embargo {auto_pe} барів"
        )
    else:
        print("    ⚠ замало успішних варіантів для CSCV")

    # ── 6. Sensitivity головного параметра ────────────────────────────────────
    print("\n[5] ЧУТЛИВІСТЬ (OOS-регіон: уся історія мінус перше train-вікно)")
    oos_region = common.iloc[args.train :]
    l1_o, l2_o = _extract_leg_df(oos_region, 1), _extract_leg_df(oos_region, 2)
    fo1 = _fund_slice(f1, oos_region.index[0], oos_region.index[-1])
    fo2 = _fund_slice(f2, oos_region.index[0], oos_region.index[-1])
    sens_param = "entry_z"
    grid = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    sharpes: list[float] = []
    sens_rows: list[dict] = []
    for v in grid:
        p = dict(params)
        p[sens_param] = v
        try:
            r = run_pairs_backtest(
                l1_o,
                l2_o,
                type(strategy)(**p),
                fo1,
                fo2,
                position_pct=args.position_pct,
                cost=cost,
                maker_execution=args.maker,
            )
            sharpes.append(float(r.metrics.sharpe_hourly))
            sens_rows.append(
                {
                    sens_param: v,
                    "sharpe_hourly": float(r.metrics.sharpe_hourly),
                    "return": float(r.metrics.total_return),
                    "n_trades": int(r.metrics.n_trades),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sensitivity %s=%s: %s", sens_param, v, exc)
    smoothness = float("nan")
    if len(sharpes) >= 3:
        arr = np.array(sharpes, dtype=float)
        span = max(abs(arr).max(), 1e-9)
        mean_abs_step = float(np.abs(np.diff(arr)).mean())
        smoothness = 1.0 - min(mean_abs_step / span, 1.0)
        for r in sens_rows:
            print(
                f"    {sens_param}={r[sens_param]:<5} SRh {r['sharpe_hourly']:+.4f} | ret {r['return']:+.2%} | угод {r['n_trades']}"
            )
        print(f"    smoothness {smoothness:.3f} (поріг > 0.30) | розкид SRh {arr.min():+.4f}…{arr.max():+.4f}")

    # ── 7. Stress + benchmark ─────────────────────────────────────────────────
    print("\n[6] STRESS / BENCHMARK")
    stress_crash = stress_liq = None
    if bt.equity is not None and len(bt.equity) > 10:
        ret = bt.equity.pct_change().dropna()
        sdf = stress_report(ret)
        print(sdf.to_string(float_format=lambda x: f"{x:+.4f}"))
        if "crash" in sdf.index:
            stress_crash = abs(float(sdf.loc["crash", "max_drawdown"]))
        if "liquidity" in sdf.index:
            stress_liq = abs(float(sdf.loc["liquidity", "max_drawdown"]))
    bh1 = buy_and_hold_sharpe(df1)
    bh2 = buy_and_hold_sharpe(df2)
    print(f"    Buy&Hold Sharpe: {leg1} {bh1:.3f} | {leg2} {bh2:.3f} | стратегія {m.sharpe:.3f}")

    # ── 8. Вердикт ────────────────────────────────────────────────────────────
    label, reasons = evaluate_pair_wf_gate(
        pos_frac,
        len(windows),
        stress_crash_max_dd=stress_crash,
        stress_liquidity_max_dd=stress_liq,
        backtest_max_dd=bt_max_dd,
    )
    # PBO — окремий критерій pairs_gate (PBO_MAX=0.50); WF-гейт його не включає.
    from scalper_hft.validation.pairs_gate import MIN_TRADES, PBO_MAX, WF_POS_FRAC_MIN

    if pbo == pbo and pbo > PBO_MAX:
        reasons = [*reasons, f"PBO={pbo:.3f}>{PBO_MAX:.2f}"]
    if int(m.n_trades) < MIN_TRADES:
        reasons = [*reasons, f"trades={int(m.n_trades)}<{MIN_TRADES}"]
    if smoothness == smoothness and smoothness < 0.30:
        reasons = [*reasons, f"smoothness={smoothness:.2f}<0.30"]
    label = "PASS" if not reasons else "FAIL"

    print("\n" + "═" * 78)
    print(
        f"КРИТЕРІЇ pairs-гейта: WF_pos >= {WF_POS_FRAC_MIN:.0%} | PBO < {PBO_MAX:.2f} | "
        f"trades >= {MIN_TRADES} | stress/baseline DD | smoothness > 0.30"
    )
    print(
        f"ФАКТ: WF_pos {pos_frac:.0%} | PBO {pbo:.3f} | trades {int(m.n_trades)} | "
        f"maxDD {bt_max_dd:.2%} | smoothness {smoothness:.3f} | DSR {dsr:.4f}"
    )
    print(f"ВЕРДИКТ: {label}" + (f"\n  причини: {'; '.join(reasons)}" if reasons else "  (усі критерії виконано)"))
    print("═" * 78)

    if not args.no_record:
        from scalper_hft.validation.verdict_store import record_pair_verdict

        record_pair_verdict(args.strategy, leg1, leg2, args.interval, label, "; ".join(reasons))
        print(f"Вердикт записано у verdict_store ({args.strategy} {leg1}/{leg2} {args.interval} {label})")

    # ── 9. Markdown-звіт ──────────────────────────────────────────────────────
    out = (
        Path(args.out)
        if args.out
        else (
            Path("docs/reports") / f"pair_audit_{args.strategy}_{leg1}_{leg2}_{args.interval}_"
            f"{pd.Timestamp.utcnow():%Y-%m-%d_%H%M}.md"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Pair audit: {args.strategy} {leg1}/{leg2} {args.interval}",
        "",
        f"- Дата (UTC): {pd.Timestamp.utcnow():%Y-%m-%d %H:%M}",
        f"- Дані: LIVE `DATA_EXCHANGE={getattr(settings, 'data_exchange', '?')}`, {args.days} днів, "
        f"{len(common)} спільних барів ({common.index[0]} … {common.index[-1]})",
        f"- Витрати: maker {settings.maker_fee:.4%} / taker {settings.taker_fee:.4%}, "
        f"slippage {settings.slippage_bps} bps, execution={'maker' if args.maker else 'taker'}",
        f"- Параметри: `{params}`",
        f"- Рантайм: {time.monotonic() - t_start:.0f} с",
        "",
        "## 1. Повний бектест (IS, інфо)",
        "",
        "| метрика | значення |",
        "|---|---|",
        f"| total_return | {m.total_return:+.2%} |",
        f"| Sharpe (річний) | {m.sharpe:.3f} |",
        f"| Sharpe (годинний) | {m.sharpe_hourly:+.4f} |",
        f"| max drawdown | {m.max_drawdown:.2%} |",
        f"| угоди | {int(m.n_trades)} |",
        f"| profit factor | {m.profit_factor:.3f} |",
        f"| win rate | {m.win_rate:.1%} |",
        f"| funding PnL | {bt.funding_pnl:+.3f}% |",
        f"| експозиція | {m.exposure:.1%} |",
        "",
        "## 2. Walk-forward",
        "",
        f"- вікон: **{len(windows)}** (train={args.train}, test={args.test})",
        f"- avg IS SRh: **{is_sh.mean():+.4f}**, avg OOS SRh: **{oos_sh.mean():+.4f}**",
        f"- позитивних OOS: **{pos_frac:.0%}** (поріг ≥ {WF_POS_FRAC_MIN:.0%})",
        f"- OOS угод: {n_trades_oos}",
        "",
        "| # | OOS від | OOS до | IS SRh | OOS SRh | OOS ret | OOS maxDD | угод |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for w in windows:
        lines.append(
            f"| {w['idx']} | {w['test_start'][:16]} | {w['test_end'][:16]} | {w['is_sharpe']:+.3f} | "
            f"{w['oos_sharpe']:+.3f} | {w['oos_return']:+.2%} | {w['oos_maxdd']:.2%} | {w['n_trades']} |"
        )
    lines += [
        "",
        "## 3. Deflated Sharpe (на конкатенованих OOS)",
        "",
        f"- DSR **{dsr:.4f}** (n_trials={n_trials}, OOS-барів={len(oos_cat)}) — поріг 0.95",
        "",
        "## 4. CSCV / PBO",
        "",
        f"- варіантів: {len(rows)}, блоків: {args.blocks}",
        f"- **PBO {pbo:.3f}** (поріг < {PBO_MAX:.2f})",
        (
            f"- OOS Sharpe IS-кращих: медіана {np.median(pbo_res.is_best_oos_sharpes):+.4f}, "
            f"частка < 0: {(pbo_res.is_best_oos_sharpes < 0).mean():.0%}"
            if pbo_res is not None
            else "- PBO не пораховано"
        ),
        "",
        "## 5. Sensitivity",
        "",
        f"- параметр `{sens_param}`, smoothness **{smoothness:.3f}** (поріг > 0.30)",
        "",
        "| entry_z | SRh | return | угод |",
        "|---:|---:|---:|---:|",
    ]
    for r in sens_rows:
        lines.append(f"| {r[sens_param]} | {r['sharpe_hourly']:+.4f} | {r['return']:+.2%} | {r['n_trades']} |")
    lines += [
        "",
        "## 6. Stress / benchmark",
        "",
        f"- stress crash maxDD: {stress_crash if stress_crash is None else f'{stress_crash:.2%}'}",
        f"- stress liquidity maxDD: {stress_liq if stress_liq is None else f'{stress_liq:.2%}'}",
        f"- baseline maxDD: {bt_max_dd:.2%}",
        f"- Buy & Hold Sharpe: {leg1} {bh1:.3f}, {leg2} {bh2:.3f}; стратегія {m.sharpe:.3f}",
        "",
        "## 7. Вердикт",
        "",
        f"**{label}**" + (f" — {'; '.join(reasons)}" if reasons else " — усі критерії виконано"),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown-звіт: {out}")

    summary = {
        "pair": f"{leg1}/{leg2}",
        "interval": args.interval,
        "days": args.days,
        "maker": bool(args.maker),
        "params": params,
        "n_common_bars": int(len(common)),
        "bt_total_return": float(m.total_return),
        "bt_sharpe": float(m.sharpe),
        "bt_sharpe_hourly": float(m.sharpe_hourly),
        "bt_max_dd": float(m.max_drawdown),
        "bt_n_trades": int(m.n_trades),
        "bt_profit_factor": float(m.profit_factor),
        "bt_win_rate": float(m.win_rate),
        "funding_pnl": float(bt.funding_pnl),
        "n_windows": len(windows),
        "avg_is_sharpe_hourly": float(is_sh.mean()),
        "avg_oos_sharpe_hourly": float(oos_sh.mean()),
        "oos_pos_frac": pos_frac,
        "dsr": float(dsr),
        "n_trials": int(n_trials),
        "pbo": float(pbo),
        "smoothness": float(smoothness),
        "stress_crash_max_dd": stress_crash,
        "stress_liquidity_max_dd": stress_liq,
        "benchmark_leg1_sharpe": float(bh1),
        "benchmark_leg2_sharpe": float(bh2),
        "windows": windows,
        "verdict": label,
        "reasons": reasons,
        "report": str(out),
    }
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {json_out}")
    return 0 if label == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
