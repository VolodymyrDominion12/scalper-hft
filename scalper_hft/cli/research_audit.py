"""Research CLI: overfit-audit, CSCV, stress, аналітика."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scalper_hft.cli._common import (
    _apply_use_kalman,
    _enqueue_job,
    _param_combinations,
    fail,
)
from scalper_hft.config import get_settings


def _require_audit_mode_prereqs(mode: str, explicit_holdout: bool) -> None:
    """Fail-fast на передумовах final-режиму (замість падіння всередині аудиту).

    Раніше `overfit` завжди стартував у mode="final" (хардкод), а final вимагає
    HOLDOUT_PCT>0 і OOS_ENFORCE_BURN=true. На дефолтному .env це давало
    миттєвий `RuntimeError: final mode: потрібен HOLDOUT_PCT>0...` після повного
    прогону — 4 такі джоби згоріли, і жодного реального вердикту не з'явилось.
    """
    if mode != "final":
        return
    settings = get_settings()
    if float(getattr(settings, "enforce_holdout_pct", 0.0)) <= 0 and not explicit_holdout:
        fail(
            "final-аудит вимагає HOLDOUT_PCT>0 у .env (останні N% історії лишаються "
            "для сліпого тесту) або явного --explicit-holdout. "
            "Для дослідницького прогону використовуйте --audit-mode exploratory (дефолт)."
        )
    if not bool(getattr(settings, "enforce_oos_burn", False)):
        fail(
            "final-аудит вимагає OOS_ENFORCE_BURN=true у .env (fail-closed на повторне "
            "використання того самого OOS-вікна)."
        )


def cmd_overfit(args: argparse.Namespace) -> None:
    """Повний аудит на перенавчання: WF + sensitivity (OOS) + DSR (OOS) + CSCV PBO + вердикт."""
    if getattr(args, "enqueue", False):
        from scalper_hft.research.job_handlers import payload_from_overfit_cli

        _enqueue_job("overfit", payload_from_overfit_cli(args))
        return
    from scalper_hft.application import RunCellAudit, run_cell_audit
    from scalper_hft.validation.cell_audit import AuditMode, cell_verdict

    print("═" * 60)
    print(f"AUDIT: стратегія {args.strategy}, {args.symbol} {args.interval}, {args.days} днів")
    print("═" * 60)

    audit_mode: AuditMode = getattr(args, "audit_mode", "exploratory") or "exploratory"
    explicit_holdout = bool(getattr(args, "explicit_holdout", False))
    _require_audit_mode_prereqs(audit_mode, explicit_holdout)
    audit = run_cell_audit(
        RunCellAudit(
            strategy=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            days=args.days,
            mode=audit_mode,
            train_bars=args.train,
            test_bars=args.test,
            with_cscv=True,
            explicit_holdout=explicit_holdout,
            strategy_params=_apply_use_kalman(args, args.param_dict),
            purge_bars=getattr(args, "purge_bars", None),
            embargo_bars=getattr(args, "embargo_bars", None),
            n_trials_floor=int(getattr(args, "trials", 0)) or None,
        )
    )
    if audit.status != "ok":
        fail(f"Аудит не вдався: {audit.error}")

    print("\n[1] WALK-FORWARD")
    print(
        f"  вікон: {audit.n_windows} | avg IS Sharpe: {audit.avg_is_sharpe:+.3f} | "
        f"avg OOS Sharpe: {audit.avg_oos_sharpe:+.3f}"
    )
    print(
        f"  частка вікон OOS>0: {audit.oos_pos_frac:.0%} | деградація IS→OOS: {audit.degradation:.1%} | "
        f"OOS угод: {audit.n_trades_oos}"
    )
    for w in audit.windows:
        print(
            f"  [{w['window_idx']}] IS {w['train_start']}:{w['train_end']} → OOS {w['test_start']}:{w['test_end']} "
            f"| IS SR {w['is_sharpe']:+.2f} | OOS SR {w['oos_sharpe']:+.2f} | ret {w['oos_return']:+.2%} "
            f"| {w['n_trades']} угод"
        )

    print("\n[2] ЧУТЛИВІСТЬ ДО ПАРАМЕТРА (сітка на OOS-регіоні)")
    if audit.sens_param:
        sm = f"{audit.smoothness:.2f}" if audit.smoothness is not None else "nan"
        print(f"  параметр: {audit.sens_param} | smoothness: {sm} | точок: {audit.sens_n}")
    if audit.sens_error:
        print(f"  помилка sensitivity: {audit.sens_error}")

    print("\n[3] DEFLATED SHARPE (на конкатенованих OOS-дохідностях)")
    dsr_s = f"{audit.dsr:.3f}" if audit.dsr is not None else "nan"
    print(f"  DSR: {dsr_s} | trials: {audit.n_trials_dsr} | DSR > 0.95 → edge статистично значущий")

    print("\n[4] CSCV PBO (Bailey–López de Prado)")
    pbo_s = f"{audit.pbo:.3f}" if audit.pbo is not None else "n/a"
    print(f"  PBO: {pbo_s} | PBO < 0.5 → перенавчання малоймовірне")

    print("\n[5] ПОВНИЙ БЕКТЕСТ (інфо, IS-забруднений)")
    print(
        f"  return {audit.bt_total_return:+.2%} | Sharpe {audit.bt_sharpe:.2f} | maxDD {audit.bt_max_dd:.2%} | "
        f"угод {audit.bt_n_trades} | PF {audit.bt_profit_factor:.2f} | win {audit.bt_win_rate:.0%}"
    )

    print("\n[6] BENCHMARK / HOLDOUT / РОЗШИРЕНІ ТЕСТИ")
    bench_s = f"{audit.benchmark_sharpe:.3f}" if audit.benchmark_sharpe is not None else "n/a"
    ho_s = f"{audit.holdout_sharpe:.3f}" if audit.holdout_sharpe is not None else "n/a"
    print(f"  Buy&Hold Sharpe: {bench_s} | Holdout Sharpe: {ho_s}")
    if audit.quintile_spearman is not None:
        q_flag = "PASS" if audit.quintile_pass else "FAIL"
        print(f"  Quintile: {q_flag} | ρ={audit.quintile_spearman:+.3f}")
    if audit.time_decay_pass is not None:
        print(f"  Time-decay: {'PASS' if audit.time_decay_pass else 'FAIL/WARN'}")
    if audit.stress_pass is not None:
        print(f"  Stress: {'PASS' if audit.stress_pass else 'FAIL'}")

    label, reasons = cell_verdict(audit, mode=audit_mode)
    print("\n" + "═" * 60)
    print(f"ВЕРДИКТ: {label}" + (f"\n  причини: {reasons}" if reasons else ""))

    # Журнал вердиктів — hard-гейт для paper/live (live/audit_gate.py; лише PASS)
    from scalper_hft.validation.verdict_store import record_verdict

    if label in {"PASS", "EXPLORATORY_PASS", "FAIL"}:
        record_verdict(args.strategy, args.symbol, args.interval, label, reasons)

    # Авто pair-вердикт для pairs-стратегій (раніше — лише ручний запис).
    # Якщо стратегія multi-symbol (pairs_arb/sparse_basket/funding_arb) і задано
    # --leg1/--leg2, проганяємо pairs walk-forward і записуємо pair-вердикт за
    # критеріями pairs_gate (WF positive frac, PBO, n_trades) — щоб pairs_runner
    # міг стартувати без ручного запису.
    _maybe_record_pair_verdict(args)


def _maybe_record_pair_verdict(args: argparse.Namespace) -> None:
    """Авто pair-вердикт для multi-symbol стратегій з --leg1/--leg2."""
    leg1 = getattr(args, "leg1", None)
    leg2 = getattr(args, "leg2", None)
    if not leg1 or not leg2:
        return
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.sweep import MULTI_SYMBOL_STRATEGIES

    strategy = get_strategy(args.strategy, **_apply_use_kalman(args, args.param_dict))
    if args.strategy not in MULTI_SYMBOL_STRATEGIES:
        return  # не pairs — pair-вердикт не потрібен
    from scalper_hft.backtest.pairs import run_pairs_walk_forward
    from scalper_hft.cli import _load_klines
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.validation.pairs_gate import evaluate_pair_wf_gate
    from scalper_hft.validation.verdict_store import record_pair_verdict

    df1 = _load_klines(
        leg1, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    df2 = _load_klines(
        leg2, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    f1 = download_funding(leg1, args.days) if strategy.needs_funding else None
    f2 = download_funding(leg2, args.days) if strategy.needs_funding else None
    try:
        wf = run_pairs_walk_forward(
            df1,
            df2,
            strategy,
            f1,
            f2,
            train_bars=args.train,
            test_bars=args.test,
            position_pct=getattr(args, "position_pct", None) or 0.3,
            maker_execution=getattr(args, "maker", False),
        )
    except ValueError as exc:
        print(f"\n[PAIR-вердикт] пропущено: {exc}")
        return

    pos_frac = float(wf["positive_windows"])
    n_windows = int(wf["n_windows"])
    # PBO — опційно (дорого); залишимо як майбутнє розширення (CSCV на pairs)

    backtest_max_dd = None
    stress_crash = None
    stress_liq = None
    try:
        from scalper_hft.backtest.pairs import run_pairs_backtest

        bt = run_pairs_backtest(
            df1,
            df2,
            strategy,
            f1,
            f2,
            position_pct=getattr(args, "position_pct", None) or 0.3,
            maker_execution=getattr(args, "maker", False),
        )
        backtest_max_dd = abs(float(bt.metrics.max_drawdown))
        if bt.equity is not None and len(bt.equity) > 10:
            ret = bt.equity.pct_change().dropna()
            from scalper_hft.validation.stress import stress_report

            stress_df = stress_report(ret)
            stress_crash = abs(float(stress_df.loc["crash", "max_drawdown"])) if "crash" in stress_df.index else None
            stress_liq = (
                abs(float(stress_df.loc["liquidity", "max_drawdown"])) if "liquidity" in stress_df.index else None
            )
    except Exception as exc:
        print(f"  [PAIR-вердикт] помилка стрес-тесту: {exc}")

    label, reasons = evaluate_pair_wf_gate(
        pos_frac,
        n_windows,
        stress_crash_max_dd=stress_crash,
        stress_liquidity_max_dd=stress_liq,
        backtest_max_dd=backtest_max_dd,
    )
    print(
        "\n[PAIR-вердикт] "
        + f"{leg1}/{leg2} {args.interval}: {label}"
        + (f" ({'; '.join(reasons)})" if reasons else "")
    )
    record_pair_verdict(args.strategy, leg1, leg2, args.interval, label, "; ".join(reasons))


def cmd_ml(args: argparse.Namespace) -> None:
    """Walk-forward ML: Triple-Barrier + LightGBM + AFML sample weights."""
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.ml.trainer import train_from_ohlcv

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    trades = None
    if args.trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)

    mode = getattr(args, "mode", "triple_barrier")
    pt = float(getattr(args, "pt", 1.0))
    sl = float(getattr(args, "sl", 1.0))
    holding = int(getattr(args, "holding", 10))
    decay = float(getattr(args, "decay", 0.9))
    frac_d = float(getattr(args, "frac_d", 0.4))
    no_frac = bool(getattr(args, "no_frac_diff", False))
    add_hmm = bool(getattr(args, "hmm", False))
    add_garch = bool(getattr(args, "garch", False))
    hmm_states = int(getattr(args, "hmm_states", 3))

    try:
        res = train_from_ohlcv(
            df=df,
            train_size=args.train,
            test_size=args.test,
            mode=mode,
            pt=pt,
            sl=sl,
            holding_bars=holding,
            decay=decay,
            frac_d=frac_d,
            add_frac_diff=not no_frac,
            trades=trades,
            add_hmm=add_hmm,
            add_garch=add_garch,
            hmm_states=hmm_states,
        )
    except ValueError as e:
        fail("ML тренування: %s", e)

    print("\n" + res.summary())

    if res.feature_importance is not None:
        print("\nТоп-10 фіч (gain):")
        print(res.feature_importance.head(10).to_string())


def cmd_report(args: argparse.Namespace) -> None:
    """Markdown-звіт: бектест + WF + sensitivity + deflated Sharpe → docs/reports/."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest as run_backtest
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.benchmark import buy_and_hold_sharpe
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
    from scalper_hft.validation.sensitivity import parameter_sensitivity
    from scalper_hft.validation.walk_forward import run_walk_forward

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel.from_settings(settings, df=df)
    bench_sr = buy_and_hold_sharpe(df)
    from scalper_hft.cli._common import _load_optional_streams

    trades, funding = _load_optional_streams(strategy, args.symbol, args.days)
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)

    res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
    wf = run_walk_forward(
        df,
        strategy,
        train_bars=args.train,
        test_bars=args.test,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
        collect_oos_returns=True,
    )
    # DSR на конкатенованих OOS-дохідностях walk-forward (як у audit_cell):
    # full-sample equity забруднена IS-вікнами і системно завищує DSR.
    oos_ret = wf.oos_returns.dropna() if wf.oos_returns is not None else pd.Series(dtype=float)
    n_trials = estimate_n_trials(_param_combinations(strategy), args.trials or 1)
    dsr = deflated_sharpe_ratio(oos_ret.values, n_trials=n_trials) if len(oos_ret) >= 2 else 0.0

    sens_md = ""
    if strategy.param_space:
        pname = next(iter(strategy.param_space))
        lo, hi, step = strategy.param_space[pname]
        values = [lo + i * step for i in range(int((hi - lo) / step) + 1)][:15]
        try:
            sens = parameter_sensitivity(df, strategy, pname, values, cost=cost, trades=trades, funding=funding)
            sens_md = f"\n## Sensitivity ({pname})\n\nsmoothness = {sens.smoothness:.3f}\n\n" + sens.grid.to_markdown(
                index=False
            )
        except Exception as exc:  # noqa: BLE001
            sens_md = f"\n## Sensitivity\n\nпомилка: {exc}"

    # ── Quintile study (Narang гл. 9) ────────────────────────────────────────
    _QUINTILE_SPEARMAN_MIN = 0.7
    quintile_md = ""
    q_res = None
    try:
        from scalper_hft.validation.quintile import quintile_spread_study

        signals = strategy.generate_signals(df, trades=trades, funding=funding)
        fwd_ret = df["close"].pct_change().shift(-1).fillna(0.0)
        if signals.abs().sum() > 5:
            q_res = quintile_spread_study(signals.astype(float), fwd_ret)
            quintile_md = f"\n## Quintile Study (монотонність сигналу, Narang гл. 9)\n\n```\n{q_res.summary()}\n```\n"
            if q_res.monotonic and abs(q_res.spearman) >= _QUINTILE_SPEARMAN_MIN:
                quintile_md += f"\n✅ QUINTILE PASS: Spearman ρ={q_res.spearman:+.3f} ≥ {_QUINTILE_SPEARMAN_MIN}\n"
            else:
                quintile_md += (
                    f"\n⚠ QUINTILE FAIL: Spearman ρ={q_res.spearman:+.3f}, "
                    f"monotonic={q_res.monotonic} — сигнал не монотонний\n"
                )
    except Exception as exc:  # noqa: BLE001
        quintile_md = f"\n## Quintile Study\n\nпомилка: {exc}\n"

    # ── Time-decay test (Narang гл. 9) ───────────────────────────────────────
    decay_md = ""
    td_res = None
    try:
        from scalper_hft.validation.time_decay import time_decay_test

        td_res = time_decay_test(
            df,
            strategy,
            max_lag=3,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
        )
        decay_md = f"\n## Time-Decay Test (лаг входу, Narang гл. 9)\n\n```\n{td_res.summary()}\n```\n"
        if len(td_res.sharpes) >= 2:
            lag0, lag1 = td_res.sharpes[0], td_res.sharpes[1]
            if lag0 > 0 and lag1 < 0:
                decay_md += "\n⚠ TIME-DECAY FAIL: Sharpe вмирає за 1 бар — edge execution-dependent!\n"
            elif lag0 > 0 and lag1 < lag0 * 0.5:
                decay_md += f"\n⚠ TIME-DECAY WARN: деградація {1 - lag1 / lag0:.0%} за 1 бар — execution-sensitive\n"
            else:
                decay_md += f"\n✅ TIME-DECAY PASS: Sharpe зберігається при lag=1 ({lag1:+.3f})\n"
    except Exception as exc:  # noqa: BLE001
        decay_md = f"\n## Time-Decay Test\n\nпомилка: {exc}\n"

    # ── Stress: cost_concentration + 4-scenario (Narang гл. 4, 10) ───────────
    stress_md = ""
    stress_report_df = None
    try:
        from scalper_hft.validation.stress import cost_concentration_stress, stress_report

        # 1) Традиційний cost-concentration стрес (fees×2, drop top-5)
        stress_tbl = cost_concentration_stress(
            df,
            strategy,
            cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
            baseline=res,
        )
        stress_md = "\n## Стрес (fees×2, slippage×2, без топ-5 угод)\n\n" + stress_tbl.to_markdown() + "\n"

        # 2) 4-сценарний стрес (crash / liquidity / vol_spike / funding_shock)
        if res.bar_returns is not None and len(res.bar_returns) > 10:
            stress_report_df = stress_report(res.bar_returns)
            stress_md += "\n## Стрес-сценарії (crash / liquidity / vol_spike / funding_shock)\n\n"
            stress_md += stress_report_df.to_markdown() + "\n"
            # pass/fail: maxDD при crash ≤ baseline × 2.5
            baseline_dd = abs(float(res.metrics.max_drawdown))
            crash_dd = (
                abs(float(stress_report_df.loc["crash", "max_drawdown"])) if "crash" in stress_report_df.index else 0.0
            )
            liq_dd = (
                abs(float(stress_report_df.loc["liquidity", "max_drawdown"]))
                if "liquidity" in stress_report_df.index
                else 0.0
            )
            if baseline_dd > 0 and crash_dd > baseline_dd * 2.5:
                stress_md += f"\n⚠ STRESS WARN: crash maxDD={crash_dd:.1%} > {baseline_dd * 2.5:.1%} (baseline×2.5)\n"
            elif baseline_dd > 0 and liq_dd > baseline_dd * 4.0:
                stress_md += f"\n⚠ STRESS WARN: liquidity maxDD={liq_dd:.1%} > {baseline_dd * 4.0:.1%} (baseline×4)\n"
            else:
                stress_md += "\n✅ STRESS PASS: всі сценарії в межах допустимих порогів\n"
    except Exception as exc:  # noqa: BLE001
        stress_md = f"\n## Стрес\n\nпомилка: {exc}\n"

    md = f"""# Звіт: {args.strategy} · {args.symbol} · {args.interval}

Дані: {len(df)} барів ({df.index[0]} … {df.index[-1]}), {args.days} днів.
Комісії: maker {cost.maker_fee:.4%}, taker {cost.taker_fee:.4%}, slippage {cost.slippage_frac:.4%}.

## Бектест

```
{res.summary()}
```

## Walk-Forward

```
{wf.summary()}
```

## Deflated Sharpe (на OOS-дохідностях WF, не full-sample)

- raw Sharpe (full-sample, IS-забруднений): {res.metrics.sharpe:.3f}
- Buy & Hold Sharpe: {bench_sr:.3f}
- OOS спостережень: {len(oos_ret)}
- trials: {n_trials}
- **DSR: {dsr:.3f}** {"✅ edge значущий" if dsr > 0.95 else "⚠ edge не підтверджено"}
{sens_md}{quintile_md}{decay_md}{stress_md}
## Висновок

- OOS Sharpe: {wf.avg_oos_sharpe:.3f} ({wf.positive_windows_frac:.0%} вікон > 0)
- DSR: {dsr:.3f}
- {"Стратегія готова до paper trading" if wf.avg_oos_sharpe > 0.3 and dsr > 0.9 else "Стратегія потребує доопрацювання"}
"""
    out_dir = Path("docs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.strategy}_{args.symbol}_{args.interval}.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)

    # ── Термінальний підсумок: pass/fail для quintile + time-decay + stress ───
    print("═" * 60)
    print("ПІДСУМОК ДОДАТКОВИХ ТЕСТІВ (Narang гл. 9, 4, 10)")
    print("─" * 60)

    # Quintile
    if q_res is not None:
        q_pass = q_res.monotonic and abs(q_res.spearman) >= _QUINTILE_SPEARMAN_MIN
        flag = "✅ PASS" if q_pass else "⚠ FAIL"
        print(f"  Quintile Study:   {flag} | Spearman ρ={q_res.spearman:+.3f} | monotonic={q_res.monotonic}")
    else:
        print("  Quintile Study:   — (недостатньо сигналів або помилка)")

    # Time-decay
    if td_res is not None and len(td_res.sharpes) >= 2:
        lag0, lag1 = td_res.sharpes[0], td_res.sharpes[1]
        if lag0 > 0 and lag1 < 0:
            td_flag = "⚠ FAIL"
        elif lag0 > 0 and lag1 < lag0 * 0.5:
            td_flag = "⚠ WARN"
        else:
            td_flag = "✅ PASS"
        sharpes_str = " | ".join(f"lag{i}={s:+.3f}" for i, s in zip(td_res.lags, td_res.sharpes))
        print(f"  Time-Decay Test:  {td_flag} | {sharpes_str}")
    else:
        print("  Time-Decay Test:  — (помилка або недостатньо даних)")

    # Stress
    if stress_report_df is not None and res.bar_returns is not None:
        baseline_dd = abs(float(res.metrics.max_drawdown))
        crash_dd = (
            abs(float(stress_report_df.loc["crash", "max_drawdown"])) if "crash" in stress_report_df.index else 0.0
        )
        liq_dd = (
            abs(float(stress_report_df.loc["liquidity", "max_drawdown"]))
            if "liquidity" in stress_report_df.index
            else 0.0
        )
        stress_ok = baseline_dd <= 0 or (crash_dd <= baseline_dd * 2.5 and liq_dd <= baseline_dd * 4.0)
        st_flag = "✅ PASS" if stress_ok else "⚠ WARN"
        print(f"  Stress (4 сцен.): {st_flag} | crash maxDD={crash_dd:.1%} | liquidity maxDD={liq_dd:.1%}")
    else:
        print("  Stress (4 сцен.): — (помилка або немає bar_returns)")

    print("─" * 60)
    print(f"Звіт збережено: {out_path}")


def cmd_cscv(args: argparse.Namespace) -> None:
    """PBO через Combinatorial Purged CV (López de Prado)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.cscv import pbo_cscv, variant_returns

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    from scalper_hft.cli._common import _load_optional_streams

    trades, funding = _load_optional_streams(strategy, args.symbol, args.days)
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)

    print(f"Генерація {args.variants} випадкових варіантів параметрів {args.strategy}...")
    returns = variant_returns(
        df,
        strategy,
        n_variants=args.variants,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
    )
    # AFML-дисципліна за замовчуванням: purge/embargo = 1% довжини ряду
    # (мін. 1 бар); явний 0 через --purge-bars/--embargo-bars вимикає.
    n_bars = int(returns.shape[1])
    auto_pe = max(1, n_bars // 100)
    purge_bars = auto_pe if getattr(args, "purge_bars", None) is None else int(args.purge_bars)
    embargo_bars = auto_pe if getattr(args, "embargo_bars", None) is None else int(args.embargo_bars)
    res = pbo_cscv(
        returns,
        n_blocks=args.blocks,
        threshold=0.0,
        max_combos=args.max_combos,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    print("\n" + res.summary())
    print(f"(purge={purge_bars}, embargo={embargo_bars} барів навколо test-блоків)")


def cmd_experiments(args: argparse.Namespace) -> None:
    """Показати каталог експериментів (val → OOS gate)."""
    from pathlib import Path

    from scalper_hft.validation.experiments import load_catalog

    path = Path(args.catalog) if args.catalog else None
    rows = load_catalog(path)
    if not rows:
        print("Каталог порожній.")
        return
    for exp in rows:
        base = exp.baseline_id or "—"
        print(
            f"{exp.id}\t{exp.strategy}\tval={exp.val_start}..{exp.val_end}\t"
            f"test={exp.test_start}..{exp.test_end}\tbaseline={base}\t{exp.hypothesis}"
        )


def cmd_cohort(args: argparse.Namespace) -> None:
    """Cohort analysis: деградація edge за когортами угод (Predictive Marketing)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest as run_backtest
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.cohort import cohort_report

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    from scalper_hft.cli._common import _load_optional_streams

    trades, funding = _load_optional_streams(strategy, args.symbol, args.days)
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)
    res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
    print(f"\nCohort: {args.strategy} · {args.symbol} · {args.interval} · {len(res.trades)} угод\n")
    print(cohort_report(res.trades, freq=args.freq))


def cmd_lift(args: argparse.Namespace) -> None:
    """Децильний lift-аналіз фіч (uplift-концепт, Predictive Marketing Ch.2/9)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest as run_backtest
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.config import get_settings
    from scalper_hft.features.indicators import add_standard_features
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.lift import feature_lift_report, lift_summary

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    from scalper_hft.cli._common import _load_optional_streams

    trades, funding = _load_optional_streams(strategy, args.symbol, args.days)
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)
    res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)

    feats = add_standard_features(df)
    report = feature_lift_report(res.trades, feats, pnl_col="ret", n_bins=args.bins)
    if not report:
        print("Немає угод для lift-аналізу (спробуйте більше даних або іншу стратегію).")
        return
    summary = lift_summary(report, top_k=args.top)
    print(f"\nLift-аналіз фіч: {args.strategy} · {args.symbol} · {args.interval} · {len(res.trades)} угод")
    print("\nТоп-фіч за |lift| (max_abs_lift далекий від 0 = інформативна; slope = монотонність):")
    print(summary.to_string(index=False))
    if args.detail:
        for col in summary["feature"].head(args.top).tolist():
            print(f"\n— {col} —")
            print(report[col].to_string(index=False))


def cmd_featimp(args: argparse.Namespace) -> None:
    """MDI/MDA/SFI feature importance (AFML Ch.8) з purged CV."""
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.ml.feature_importance import feature_importance_report
    from scalper_hft.ml.features import build_labeled_dataset
    from scalper_hft.validation.cv import PurgedKFold

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    trades = None
    if args.trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)
    try:
        X, y, w = build_labeled_dataset(
            df,
            trades=trades,
            mode="triple_barrier",
            pt=args.pt,
            sl=args.sl,
            holding_bars=args.holding,
            decay=args.decay,
            frac_d=args.frac_d,
            add_frac_diff=not args.no_frac_diff,
        )
    except ValueError as e:
        fail("featimp: %s", e)

    def clf_factory():
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            verbosity=-1,
        )

    pkf = PurgedKFold(n_splits=args.splits, embargo_pct=args.embargo)
    print(f"\nFeature importance (AFML Ch.8): {args.symbol} {args.interval}, {len(X)} зразків, {X.shape[1]} фіч\n")
    rep = feature_importance_report(X, y, clf_factory, pkf, sample_weights=w, score="neg_log_loss")
    print(rep.round(4).to_string())
    tau = rep["pca_tau"].iloc[0] if not rep.empty else 0.0
    print(
        f"\nPCA-перевірка (weighted Kendall τ MDI vs PCA-ранг): {tau:.3f} "
        f"{'✅ патерн не випадковий (>0.8)' if tau > 0.8 else '⚠ слабка узгодженість'}"
    )


def cmd_cfi(args: argparse.Namespace) -> None:
    """Clustered Feature Importance (AFML Ch.8.5) з Purged CV."""
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.ml.clustered_importance import clustered_mda
    from scalper_hft.ml.features import build_labeled_dataset
    from scalper_hft.validation.cv import PurgedKFold

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    trades = None
    if args.trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)
    try:
        X, y, w = build_labeled_dataset(
            df,
            trades=trades,
            mode="triple_barrier",
            pt=args.pt,
            sl=args.sl,
            holding_bars=args.holding,
            decay=args.decay,
            frac_d=args.frac_d,
            add_frac_diff=not args.no_frac_diff,
        )
    except ValueError as e:
        fail("cfi: %s", e)

    def clf_factory():
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            verbosity=-1,
        )

    pkf = PurgedKFold(n_splits=args.splits, embargo_pct=args.embargo)
    print(
        f"\nClustered Feature Importance (AFML Ch.8.5): {args.symbol} {args.interval}, {len(X)} зразків, {X.shape[1]} фіч\n"
    )
    cfi_res = clustered_mda(X, y, clf_factory, pkf, sample_weights=w, max_clusters=args.max_clusters)

    print("— Важливість кластерів ознак (Clustered MDA) —")
    for cl_id, imp in cfi_res.clustered_mda.items():
        feats = cfi_res.clusters_dict.get(int(cl_id), [])
        print(f"Кластер {cl_id:2d} (важливість: {imp:+.4f}) -> {', '.join(feats)}")

    print("\n— Топ ознак за скоригованою важливістю —")
    print(cfi_res.feature_mda.sort_values(ascending=False).round(4).to_string())


def cmd_stress(args: argparse.Namespace) -> None:
    """Стрес-тестування: crash / liquidity / vol_spike / funding_shock."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest as run_backtest
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.stress import SCENARIOS, stress_report

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    from scalper_hft.cli._common import _load_optional_streams

    trades, funding = _load_optional_streams(strategy, args.symbol, args.days)
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)
    res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
    ret = res.equity.pct_change().dropna()
    scenarios = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else list(SCENARIOS)
    rep = stress_report(ret, scenarios=scenarios)
    print(f"\nСтрес-тест: {args.strategy} · {args.symbol} · {args.interval}\n")
    print(rep.round(4).to_string())
    print(
        "\n⚠ liquidity = витрати ×10; crash = найгірше вікно ×2;"
        "\n  vol_spike = волатильність ×2; funding_shock = додаткова per-bar ставка 0.1%"
    )


def cmd_capacity(args: argparse.Namespace) -> None:
    """Capacity-тест: Sharpe при масштабуванні позицій (share of wallet)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.capacity import capacity_report

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    from scalper_hft.cli._common import _load_optional_streams

    trades, funding = _load_optional_streams(strategy, args.symbol, args.days)
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)
    scales = [float(s) for s in args.scales.split(",")] if args.scales else [1.0, 2.0, 5.0, 10.0, 20.0]
    print(
        capacity_report(
            df,
            strategy,
            scales=scales,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
            is_maker=args.maker,
        )
    )


def cmd_survival(args: argparse.Namespace) -> None:
    """Survival analysis: медіанний час утримання позиції (Kaplan–Meier)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest as run_backtest
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.survival import kaplan_meier, median_survival_time, trade_durations

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    from scalper_hft.cli._common import _load_optional_streams

    trades, funding = _load_optional_streams(strategy, args.symbol, args.days)
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)
    res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
    if res.trades is None or res.trades.empty:
        print("Немає угод для survival-аналізу.")
        return
    dur = trade_durations(res.trades, freq=args.interval)
    km = kaplan_meier(dur["duration"], dur["event"], max_time=args.max_time)
    median = median_survival_time(km)
    print(f"\nSurvival analysis: {args.strategy} · {args.symbol} · {args.interval} · {len(dur)} угод")
    print(f"Медіанний час утримання: {median:.0f} барів\n")
    print(km.head(args.top).round(4).to_string(index=False))
    if args.feature:
        from scalper_hft.features.indicators import add_standard_features
        from scalper_hft.validation.survival import survival_by_feature

        feats = add_standard_features(df)
        feat = feats[args.feature].reindex(pd.to_datetime(res.trades["entry_ts"]))
        sb = survival_by_feature(res.trades, feat, n_bins=args.bins, freq=args.interval)
        print(f"\nМедіанний час утримання за бінами фічі '{args.feature}':")
        print(sb.to_string(index=False))


def cmd_time_decay(args: argparse.Namespace) -> None:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.data.research import load_research_data
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.time_decay import time_decay_test

    strategy = get_strategy(args.strategy, **args.param_dict)
    bundle = load_research_data(args.symbol, args.interval, args.days, strategy)
    settings = get_settings()
    res = time_decay_test(
        bundle.klines,
        strategy,
        max_lag=args.max_lag,
        cost=CostModel(
            maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac
        ),
        trades=bundle.trades,
        funding=bundle.funding,
        position_pct=settings.position_pct,
    )
    print(res.summary())


def cmd_quintile(args: argparse.Namespace) -> None:
    import numpy as np

    from scalper_hft.data.research import load_research_data
    from scalper_hft.validation.quintile import quintile_spread_study

    b1 = load_research_data(args.leg1, args.interval, args.days)
    b2 = load_research_data(args.leg2, args.interval, args.days)
    common = (
        b1.klines[["close"]]
        .rename(columns={"close": "l1"})
        .join(b2.klines[["close"]].rename(columns={"close": "l2"}), how="inner")
        .dropna()
    )
    ratio = np.log(common["l1"] / common["l2"])
    lb = int(args.lookback)
    z = (ratio - ratio.rolling(lb).mean()) / ratio.rolling(lb).std(ddof=0)
    fwd = -ratio.diff().shift(-1)
    print(quintile_spread_study(z, fwd).summary())


def cmd_coint_scan(args: argparse.Namespace) -> None:
    from scalper_hft.data.research import load_research_data
    from scalper_hft.validation.coint_scan import scan_pairs

    settings = get_settings()
    symbols = (args.symbols or ",".join(settings.default_symbols)).split(",")
    closes = {}
    for sym in symbols:
        closes[sym.strip()] = load_research_data(sym.strip(), args.interval, args.days).klines["close"]
    for row in scan_pairs(closes):
        print(row.summary())


def cmd_hedge_ratio(args: argparse.Namespace) -> None:
    from scalper_hft.data.research import load_research_data
    from scalper_hft.validation.hedge_ratio import compare_hedge_oos

    b1 = load_research_data(args.leg1, args.interval, args.days)
    b2 = load_research_data(args.leg2, args.interval, args.days)
    res = compare_hedge_oos(b1.klines["close"], b2.klines["close"], lookback=int(args.lookback))
    print(res.summary())


def cmd_regime_map(args: argparse.Namespace) -> None:
    """Fit емпіричної карти «режим → стратегія» (Phase 2B) і зберегти JSON.

    Карта будується на IS-періоді (--fit-days) і застосовується у
    RegimeSupervisor через `regime_map_path`. Метадані fit-вікна пишуться в
    той самий JSON — аудит no-lookahead: застосовувати карту можна лише на
    барах ПІСЛЯ meta.fit_end.
    """
    from scalper_hft.validation.regime_fit import (
        DEFAULT_FIT_STRATEGIES,
        DEFAULT_FIT_SYMBOLS,
        fit_regime_map,
        format_fit_summary,
    )
    from scalper_hft.validation.regime_map import save_regime_map

    strategies = [s.strip() for s in (args.strategies or ",".join(DEFAULT_FIT_STRATEGIES)).split(",") if s.strip()]
    symbols = [s.strip() for s in (args.symbols or ",".join(DEFAULT_FIT_SYMBOLS)).split(",") if s.strip()]

    result = fit_regime_map(
        strategies,
        symbols,
        interval=args.interval or "1h",
        fit_days=int(args.fit_days),
        base_interval=args.base or "1m",
        is_maker=not args.taker,
        min_bars=int(args.min_bars),
        hard_off_sharpe=float(args.hard_off),
        high_vol_flat=not args.no_high_vol_flat,
        best_prior_min_gap=float(args.best_prior_gap),
    )
    if not result.map.weights:
        fail("Не вдалося побудувати карту: немає жодної валідованої комірки (regime × strategy).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_regime_map(result.map, out, meta=result.meta)
    print(format_fit_summary(result))
    print(f"\nКарта: {out}")


def cmd_regime_matrix(args: argparse.Namespace) -> None:
    """Показати Sharpe per (regime × strategy) з fit-періоду (без запису карти)."""
    from scalper_hft.validation.regime_fit import fit_regime_map

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    result = fit_regime_map(
        strategies,
        symbols,
        interval=args.interval or "1h",
        fit_days=int(args.fit_days),
        base_interval=args.base or "1m",
        is_maker=not args.taker,
        min_bars=int(args.min_bars),
    )
    rows = []
    for regime in result.matrix.regimes():
        row: dict[str, object] = {"regime": regime, "policy": result.map.policy.get(regime, "")}
        for strat in result.matrix.strategies():
            cell = result.matrix.data.get(regime, {}).get(strat)
            row[strat] = round(float(cell["sharpe"]), 3) if cell else None
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False) if rows else "Матриця порожня.")
