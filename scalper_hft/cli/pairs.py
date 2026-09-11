"""Pairs-команди CLI: статистичний арбітраж перпів, портфель пар, funding arb."""

from __future__ import annotations

import argparse

from scalper_hft.cli._common import (
    _apply_use_kalman,
    _enqueue_job,
    _plot_equity,
)
from scalper_hft.config import get_settings


def cmd_pairs(args: argparse.Namespace) -> None:
    """Статистичний арбітраж пар (BTC/ETH/SOL перпи)."""
    from scalper_hft.cli import _load_klines  # call-time (patchable)

    if getattr(args, "enqueue", False):
        from scalper_hft.research.job_handlers import payload_from_pairs_cli

        _enqueue_job("pairs", payload_from_pairs_cli(args))
        return
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs import run_pairs_backtest
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.strategies import get_strategy

    leg1, leg2 = (args.leg1 or "BTCUSDT"), (args.leg2 or "ETHUSDT")
    df1 = _load_klines(
        leg1, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    df2 = _load_klines(
        leg2, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    f1 = download_funding(leg1, args.days)
    f2 = download_funding(leg2, args.days)
    strategy = get_strategy(args.strategy, **_apply_use_kalman(args, args.param_dict))
    settings = get_settings()
    cost = CostModel.from_settings(settings, df=df1)

    res = run_pairs_backtest(
        df1,
        df2,
        strategy,
        f1,
        f2,
        position_pct=args.position_pct or 0.1,
        cost=cost,
        maker_execution=args.maker,
    )
    print(f"\nПара: {leg1} / {leg2} ({args.interval}, {len(res.spread)} спільних барів)")
    print(res.summary())
    _plot_equity(res.equity, args.strategy, f"{leg1}_{leg2}")

    from scalper_hft.validation.audit_extensions import format_pairs_signal_quality, pair_frame_from_klines

    pair_df = pair_frame_from_klines(df1, df2)
    print(format_pairs_signal_quality(pair_df, strategy))

    if args.walkforward:
        from scalper_hft.backtest.pairs import run_pairs_walk_forward

        wf = run_pairs_walk_forward(
            df1,
            df2,
            strategy,
            f1,
            f2,
            train_bars=args.train,
            test_bars=args.test,
            position_pct=args.position_pct or 0.1,
            maker_execution=args.maker,
        )
        print(
            f"\nWalk-forward: {wf['n_windows']} вікон | IS SRh={wf['avg_is_sharpe']:+.3f} | "
            f"OOS SRh={wf['avg_oos_sharpe']:+.3f} | позитивних OOS: {wf['positive_windows']:.0%}"
        )


def cmd_pairs_portfolio(args: argparse.Namespace) -> None:
    """Бектест портфеля валідованих пар (XRP/BTC + BTC/ETH + LINK/BTC)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs_portfolio import run_pairs_portfolio
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.live.pairs_runner import VALIDATED_PAIRS
    from scalper_hft.strategies.pairs_arb import PairsArb

    interval = args.interval or "1h"
    settings = get_settings()
    symbols = sorted({c["leg1"] for c in VALIDATED_PAIRS} | {c["leg2"] for c in VALIDATED_PAIRS})
    data = {
        sym: _load_klines(
            sym, interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
        )
        for sym in symbols
    }
    funding = {sym: download_funding(sym, args.days) for sym in symbols}
    configs = []
    for cfg in VALIDATED_PAIRS:
        configs.append(
            {
                "leg1": cfg["leg1"],
                "leg2": cfg["leg2"],
                "strategy": PairsArb(entry_z=cfg["entry_z"], exit_z=cfg["exit_z"], lookback=cfg["lookback"]),
                "funding1": funding[cfg["leg1"]],
                "funding2": funding[cfg["leg2"]],
            }
        )
    first_df = next(iter(data.values()))
    cost = CostModel.from_settings(settings, df=first_df)
    res = run_pairs_portfolio(
        data,
        configs,
        position_pct=args.position_pct or 0.3,
        cost=cost,
        maker_execution=True,
        method=args.method,
        turnover_rate=args.turnover_rate,
        rebalance=None if args.no_rebalance else "ME",
    )
    print(res.summary())
    _plot_equity(res.equity, "pairs_portfolio", "validated")


def cmd_arb(args: argparse.Namespace) -> None:
    """Delta-neutral funding arbitrage (перп+спот): бектест + walk-forward."""
    from scalper_hft.backtest.delta_neutral import run_delta_neutral_backtest, run_dn_walk_forward
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.cli import _load_klines  # call-time (patchable)
    from scalper_hft.data.downloader import download_funding, download_spot_klines
    from scalper_hft.strategies import get_strategy

    perp = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    spot = download_spot_klines(args.symbol, args.interval, args.days)
    funding = download_funding(args.symbol, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel.from_settings(settings, df=perp)

    res = run_delta_neutral_backtest(
        perp,
        spot,
        strategy,
        funding,
        position_pct=args.position_pct or 0.1,
        cost=cost,
        maker_execution=args.maker,
    )
    print("\n" + res.summary())
    _plot_equity(res.equity, args.strategy, args.symbol)

    if args.walkforward:
        wf = run_dn_walk_forward(
            perp,
            spot,
            strategy,
            funding,
            train_bars=args.train,
            test_bars=args.test,
            position_pct=args.position_pct or 0.1,
        )
        print(
            f"\nWalk-forward: {wf['n_windows']} вікон | avg IS SRh={wf['avg_is_sharpe']:+.3f} | "
            f"avg OOS SRh={wf['avg_oos_sharpe']:+.3f} | позитивних OOS: {wf['positive_windows']:.0%}"
        )
        if wf["avg_oos_sharpe"] < 0.1:
            print("⚠  OOS слабкий — edge не підтверджено")
