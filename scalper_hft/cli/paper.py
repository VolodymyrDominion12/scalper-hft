"""Paper-команди CLI: paper/paper-run/replay/audit (paper-only, fail-closed)."""

from __future__ import annotations

import argparse
from pathlib import Path

from scalper_hft.cli._common import (
    _plot_equity,
)


def _check_directional_audit_gate(args: argparse.Namespace) -> None:
    """Fail-closed directional overfitting-гейт для paper/paper-run.

    DRY_RUN=false: безумовний гейт (require_live_audit_if_not_dry_run).
    DRY_RUN=true: опційно при `REQUIRE_AUDIT_PASS` (дефолт False — дослідницький
    режим без блокування). Pairs-стратегії тут не проходять (окремий pair-гейт).
    """
    import scalper_hft.config as _cfg

    settings = _cfg.get_settings()
    from scalper_hft.live.audit_gate import require_audit_pass, require_live_audit_if_not_dry_run

    directional = (args.strategy, [args.symbol], args.interval)
    if not settings.dry_run:
        require_live_audit_if_not_dry_run(settings, directional=directional)
        return
    if not settings.require_audit_pass:
        return
    require_audit_pass(
        args.strategy,
        [args.symbol],
        args.interval,
        max_age_days=settings.audit_max_age_days,
    )


def cmd_paper(args: argparse.Namespace) -> None:
    # call-time імпорт: тести патчать scalper_hft.config.get_settings
    import scalper_hft.config as _cfg
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.live.trader import LiveTrader, run_trader_once
    from scalper_hft.strategies import get_strategy

    if not _cfg.get_settings().dry_run:
        raise SystemExit(
            "paper — paper-only команда: при DRY_RUN=false відмова. Реальні ордери — лише через свідомий live-запуск."
        )
    _check_directional_audit_gate(args)
    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    trader = LiveTrader(strategy, args.symbol, args.interval)
    result = run_trader_once(trader, df)
    print(f"Останній бар: {df.index[-1]}")
    print(f"Дія: {result}")
    print(
        f"Капітал: {trader.account.equity:.2f} | позиції: {len(trader.account.positions)} | угод: {len(trader.account.trades)}"
    )


def cmd_paper_run(args: argparse.Namespace) -> None:
    """Циклічний paper-прогін: кілька кроків з паузою, збереження угод."""
    # call-time імпорт: тести патчать scalper_hft.config.get_settings
    import scalper_hft.config as _cfg
    from scalper_hft.live.paper_runner import PaperRunner
    from scalper_hft.strategies import get_strategy

    if not _cfg.get_settings().dry_run:
        raise SystemExit(
            "paper-run — paper-only команда: при DRY_RUN=false відмова. "
            "Реальні ордери — лише через свідомий live-запуск."
        )
    _check_directional_audit_gate(args)
    strategy = get_strategy(args.strategy, **args.param_dict)
    runner = PaperRunner(strategy, args.symbol, args.interval)
    result = runner.run(iterations=args.iterations, sleep_sec=args.sleep)
    print("\n" + result.summary())
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        eq_str = f"{result.account.equity:.2f}" if result.account else "N/A"
        send_telegram(f"Paper-run {args.strategy} {args.symbol}: {result.actions[-1]}, equity={eq_str}")


def cmd_paper_replay(args: argparse.Namespace) -> None:
    """Відтворення історії через риск-контрольованого трейдера (валідація risk-шару)."""
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.live.paper_replay import paper_replay
    from scalper_hft.strategies import get_strategy

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **args.param_dict)
    funding = None
    if strategy.needs_funding:
        funding = download_funding(args.symbol, args.days)
    result = paper_replay(df, strategy, funding=funding, position_pct=args.position_pct or 0.01)
    print("\n" + result.summary())
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        m = result.metrics
        send_telegram(
            f"📊 Paper-replay {args.strategy} {args.symbol} {args.interval}: "
            f"ret={m.total_return:+.2%}, угод={m.n_trades}, риск-блоків={len(result.risk_blocks)}, funding={result.funding_pnl:+.2f}"
        )


def cmd_paper_run_pairs(args: argparse.Namespace) -> None:
    """Циклічний paper pairs (maker, 2 ноги) або портфель валідованих пар."""

    from scalper_hft.live.pairs_runner import PairsPaperRunner, PairsPortfolioRunner
    from scalper_hft.live.release import startup_banner
    from scalper_hft.live.store import PaperStore
    from scalper_hft.strategies import get_strategy

    store = PaperStore()
    interval = args.interval or "1h"
    control_path = Path(args.control) if getattr(args, "control", None) else None
    runner: PairsPaperRunner | PairsPortfolioRunner
    if args.portfolio:
        runner = PairsPortfolioRunner(interval=interval, store=store, is_maker=True, control_path=control_path)
        extra = "portfolio"
    else:
        strategy = get_strategy(args.strategy or "pairs_arb", **args.param_dict)
        runner = PairsPaperRunner(
            args.leg1 or "XRPUSDT",
            args.leg2 or "BTCUSDT",
            interval=interval,
            strategy=strategy,
            store=store,
            is_maker=True,
            control_path=control_path,
        )
        extra = f"{args.leg1 or 'XRPUSDT'}/{args.leg2 or 'BTCUSDT'}"
    daemon = bool(getattr(args, "daemon", False))
    banner = startup_banner(mode="paper-pairs", extra=("daemon " + extra) if daemon else extra)
    print(banner)
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        send_telegram(banner)
    if daemon:
        result = runner.run(daemon=True)
    else:
        result = runner.run(iterations=args.iterations, sleep_sec=args.sleep)
    print("\n" + result.summary())
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        eq_str = f"{result.account.equity:.2f}" if result.account else "N/A"
        send_telegram(
            f"Paper pairs {result.pair}: {result.actions[-1] if result.actions else '-'} | "
            f"equity={eq_str} fill={result.n_filled}/{result.n_filled + result.n_unfilled}"
        )


def cmd_paper_replay_pairs(args: argparse.Namespace) -> None:
    """Історичний paper pairs з моделлю unfilled post-only."""
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.live.pairs_runner import replay_pairs
    from scalper_hft.live.store import PaperStore
    from scalper_hft.strategies import get_strategy

    interval = args.interval or "1h"
    leg1, leg2 = (args.leg1 or "XRPUSDT"), (args.leg2 or "BTCUSDT")
    df1 = _load_klines(
        leg1, interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    df2 = _load_klines(
        leg2, interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy or "pairs_arb", **args.param_dict)
    store = PaperStore()
    result = replay_pairs(
        leg1,
        leg2,
        df1,
        df2,
        strategy=strategy,
        store=store,
        funding1=download_funding(leg1, args.days),
        funding2=download_funding(leg2, args.days),
        is_maker=True,
        interval=interval,
    )
    print("\n" + result.summary())
    _plot_equity(result.equity, "paper_pairs", f"{leg1}_{leg2}")
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        send_telegram(result.summary())


def cmd_paper_audit(args: argparse.Namespace) -> None:
    """Phase 1: tracking error paper SQLite vs бектест + loss forensics."""

    from scalper_hft.live.store import PaperStore
    from scalper_hft.validation.paper_audit import audit_paper_store, load_equity_csv

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"Немає paper DB: {db}")
    store = PaperStore(db)
    bt_equity = load_equity_csv(Path(args.bt_equity)) if args.bt_equity else None
    audit = audit_paper_store(
        store,
        bt_equity=bt_equity,
        bt_fill_rate=args.bt_fill_rate,
        dd_mult=args.dd_mult,
    )
    print("\n" + audit.summary())
    from scalper_hft.config import get_settings
    from scalper_hft.validation.paper_audit import format_cost_hint, is_cost_hint

    slip_frac = get_settings().slippage_bps / 10_000.0
    print(format_cost_hint(is_cost_hint([], slip_frac)))
    from scalper_hft.research.session_analysis import hourly_fill_rate, session_breakdown
    from scalper_hft.validation.forensics import trades_from_paper_frames

    orders = store.all_orders()
    fills = hourly_fill_rate(orders)
    if not fills.empty:
        print("\nFill-rate by hour UTC:")
        print(fills.to_string())
    trades = trades_from_paper_frames(store.all_trades(), orders)
    sess = session_breakdown(trades)
    if not sess.empty:
        print("\nPnL by hour UTC:")
        print(sess.to_string())
    store.close()
