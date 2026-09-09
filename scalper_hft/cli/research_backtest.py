"""Research CLI: бектест, walk-forward, оптимізація."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scalper_hft.cli._common import (
    _apply_use_kalman,
    _enqueue_job,
    _merge_overlay_params,
    _plot_equity,
    _policy_from_args,
    _warn_pairs_short_interval,
    fail,
    logger,
)
from scalper_hft.config import get_settings


def cmd_backtest(args: argparse.Namespace) -> None:
    if getattr(args, "enqueue", False):
        from scalper_hft.research.job_handlers import payload_from_backtest_cli

        _enqueue_job("backtest", payload_from_backtest_cli(args))
        return
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.data.research import load_research_data
    from scalper_hft.strategies import get_strategy

    params = _apply_use_kalman(args, dict(args.param_dict))
    overlay = _policy_from_args(args)
    if overlay is not None:
        params = _merge_overlay_params(args, params, overlay)
    if getattr(args, "breakeven_gate", False):
        params["breakeven_gate"] = True
    strategy = get_strategy(args.strategy, **params)
    _warn_pairs_short_interval(args.strategy, args.interval)
    settings = get_settings()
    exchange_id = getattr(args, "exchange", settings.exchange)
    bar_type = getattr(args, "bar_type", "time")
    bundle = load_research_data(
        args.symbol,
        args.interval,
        args.days,
        strategy,
        base=getattr(args, "base", None) or "1m",
        derive=getattr(args, "derive", True),
        exchange_id=exchange_id,
    )
    df = bundle.klines
    trades = bundle.trades
    funding = bundle.funding
    if bar_type in {"dollar", "volume"}:
        from scalper_hft.data.bars import create_dollar_bars, create_volume_bars
        from scalper_hft.data.downloader import download_agg_trades

        trades = trades if trades is not None else download_agg_trades(args.symbol, args.days)
        if trades is None or trades.empty:
            fail("Немає даних aggTrades для формування барів")
        threshold = getattr(args, "bar_threshold", 100000.0)
        df = create_dollar_bars(trades, threshold) if bar_type == "dollar" else create_volume_bars(trades, threshold)
        logger.info("Згенеровано %d %s-барів", len(df), bar_type)
    if bundle.quality is not None and not bundle.quality.ok:
        logger.warning("Якість барів: %s", bundle.quality.summary())
    cost = CostModel(
        maker_fee=settings.maker_fee,
        taker_fee=settings.taker_fee,
        slippage_frac=settings.slippage_frac,
        vol_ref=float(getattr(args, "vol_ref", 0.0) or 0.0),
    )
    queue_model = None
    if getattr(args, "queue_model", False):
        from scalper_hft.backtest.micro_price import QueuePositionModel

        queue_model = QueuePositionModel()
    res = run_strategy_backtest(
        df,
        strategy,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
        overlay=overlay,
        interval=args.interval or get_settings().default_interval,
        queue_model=queue_model,
        spread_bps=float(getattr(args, "spread_bps", 2.0)),
        intrabar_exits=bool(getattr(args, "intrabar", False)),
    )
    print("\n" + res.summary())
    _plot_equity(res.equity, args.strategy, args.symbol)


def cmd_walkforward(args: argparse.Namespace) -> None:
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.cell_audit import resolve_wf_windows
    from scalper_hft.validation.walk_forward import run_walk_forward

    settings = get_settings()
    exchange_id = getattr(args, "exchange", settings.exchange)
    interval = args.interval or settings.default_interval
    df = _load_klines(
        args.symbol,
        interval,
        args.days,
        base=getattr(args, "base", None),
        derive=getattr(args, "derive", True),
        exchange_id=exchange_id,
    )
    params = _apply_use_kalman(args, args.param_dict)
    overlay = _policy_from_args(args)
    if overlay is not None:
        params = _merge_overlay_params(args, params, overlay)
    strategy = get_strategy(args.strategy, **params)
    trades = None
    if strategy.needs_trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days, exchange_id=exchange_id)
    funding = None
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)
    train, test = resolve_wf_windows(interval, args.train, args.test)
    res = run_walk_forward(
        df,
        strategy,
        train_bars=train,
        test_bars=test,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
        overlay=overlay,
        interval=interval,
        is_maker=overlay.execution == "maker" if overlay is not None else False,
    )
    print("\n" + res.summary())
    if res.avg_oos_sharpe < 0.3:
        print("\n⚠  OOS Sharpe низький — стратегія, ймовірно, не генералізує.")


def cmd_optimize(args: argparse.Namespace) -> None:
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.optimize import optimize_params

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy)
    settings = get_settings()
    # «Замкований» holdout: Optuna-підбір йде лише на research-частині
    # (перші 100-holdout_pct %); останні holdout_pct% — сліпий фінальний тест.
    from scalper_hft.validation.holdout import split_research_holdout

    df, _holdout = split_research_holdout(df, settings.enforce_holdout_pct)
    if df.empty:
        fail("після holdout-обрізу немає даних (зменшіть HOLDOUT_PCT або збільшіть --days)")
    # OOS-дисципліна: Optuna-підбір «спалює» OOS-відрізок (strategy×symbol×дати),
    # щоб повторна оптимізація на тих самих даних не могла «випадково»
    # пере-підібрати під вже спалений OOS. Прапорець OOS_ENFORCE_BURN (default off).
    from scalper_hft.validation.oos_registry import check_and_burn

    burn_ok, burn_reason = check_and_burn(
        strategy=args.strategy,
        symbol=args.symbol,
        df=df,
        days=args.days,
        purpose="optimize/optuna",
        registry_path=settings.oos_registry_path,
        enforce=settings.enforce_oos_burn,
    )
    if not burn_ok:
        fail("OOS-дисципліна: %s", burn_reason)
    res = optimize_params(
        df,
        strategy,
        n_trials=args.trials,
        position_pct=settings.position_pct,
        n_splits=args.splits,
    )
    print("\n" + res.summary())


def cmd_ml_opt(args: argparse.Namespace) -> None:
    """Оптимізація параметрів маркування Triple-Barrier для ML-стратегій."""
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.validation.optimize import optimize_ml_params

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    trades = None
    if args.trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)

    res = optimize_ml_params(
        df=df,
        n_trials=args.trials,
        n_splits=args.splits,
        embargo_pct=args.embargo,
        scoring=args.scoring,
        decay=args.decay,
        frac_d=args.frac_d,
        add_frac_diff=not args.no_frac_diff,
        trades=trades,
        sampler=args.sampler,
    )
    print("\n[ML OPTIMIZATION RESULT]")
    print(res.summary())


def cmd_regime_backtest(args: argparse.Namespace) -> None:
    """Порівняльний бектест: базові стратегії vs RegimeSupervisor.

    Виводить таблицю Sharpe/PF/WinRate/MaxDD для кожної стратегії та
    всіх трьох режимів supervisor-а, плюс attribution по ринкових режимах.

    Приклад:
        uv run python -m scalper_hft.cli regime-backtest \\
            --strategies "mean_reversion,supertrend,hmm_reversion" \\
            --symbol BTCUSDT --interval 1h --days 180
    """
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.research.regime_analysis import (
        compare_strategies_by_regime,
        regime_transition_matrix,
        supervisor_vs_baseline,
    )
    from scalper_hft.strategies import get_strategy

    settings = get_settings()
    df = _load_klines(
        args.symbol,
        args.interval,
        args.days,
        base=getattr(args, "base", None),
        derive=getattr(args, "derive", True),
    )

    cost = CostModel(
        maker_fee=settings.maker_fee,
        taker_fee=settings.taker_fee,
        slippage_frac=settings.slippage_frac,
    )

    strat_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not strat_names:
        logger.error("--strategies не вказано або порожнє")
        return

    # ── Бектест базових стратегій ────────────────────────────────────────
    from scalper_hft.backtest.engine import run_backtest

    baseline_returns: dict[str, pd.Series] = {}
    for name in strat_names:
        try:
            strat = get_strategy(name)
            result = run_backtest(df, strat, cost=cost)
            baseline_returns[name] = result.bar_returns
            print(f"  ✓ {name}: Sharpe={result.sharpe:.2f}  PF={result.profit_factor:.2f}")
        except Exception as exc:
            logger.warning("Стратегія %s: помилка бектесту: %s", name, exc)

    if not baseline_returns:
        logger.error("Жодна базова стратегія не виконалась успішно")
        return

    # ── Бектест RegimeSupervisor ─────────────────────────────────────────
    blend_modes = [args.blend_mode] if args.blend_mode != "all" else ["regime_soft", "contextual_hedge", "exp3"]
    supervisor_results: dict[str, pd.Series] = {}
    for mode in blend_modes:
        try:
            sup = get_strategy(
                "regime_supervisor",
                strategies=args.strategies,
                blend_mode=mode,
                n_hmm_states=int(args.n_hmm_states),
                hmm_fit_bars=int(args.hmm_fit_bars),
            )
            result = run_backtest(df, sup, cost=cost)
            key = f"supervisor_{mode}"
            supervisor_results[key] = result.bar_returns
            print(f"  ✓ supervisor[{mode}]: Sharpe={result.sharpe:.2f}  PF={result.profit_factor:.2f}")
        except Exception as exc:
            logger.warning("RegimeSupervisor[%s]: помилка: %s", mode, exc)

    # ── Зведена таблиця ─────────────────────────────────────────────────
    all_returns = {**baseline_returns, **supervisor_results}
    summary = supervisor_vs_baseline(
        df["close"],
        baseline_returns,
        list(supervisor_results.values())[0] if supervisor_results else pd.Series(0.0, index=df.index),
    )
    print("\n" + "=" * 70)
    print("SUPERVISOR vs BASELINE — Загальна таблиця")
    print("=" * 70)
    print(summary.to_string())

    # ── По режимах ──────────────────────────────────────────────────────
    if not args.no_regime_table:
        regime_table = compare_strategies_by_regime(
            df["close"],
            all_returns,
            n_hmm_states=int(args.n_hmm_states),
            hmm_fit_bars=int(args.hmm_fit_bars),
        )
        print("\n" + "=" * 70)
        print("SHARPE PO РЕЖИМАХ")
        print("=" * 70)
        sharpe_pivot = regime_table["sharpe"].unstack(level="regime")
        print(sharpe_pivot.to_string())

        trans = regime_transition_matrix(
            df["close"],
            n_hmm_states=int(args.n_hmm_states),
            hmm_fit_bars=int(args.hmm_fit_bars),
        )
        print("\n" + "=" * 70)
        print("МАТРИЦЯ ПЕРЕХОДІВ МІЖ РЕЖИМАМИ (рядки нормовані)")
        print("=" * 70)
        print(trans.round(3).to_string())

    # ── Збереження ──────────────────────────────────────────────────────
    if args.save:
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"regime_backtest_{args.symbol}_{args.interval}.csv"
        summary.to_csv(out_path)
        print(f"\nЗбережено: {out_path}")
