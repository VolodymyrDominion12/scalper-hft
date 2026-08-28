"""CLI для scalper-hft.

Приклади:
    python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 30
    python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 90
    python -m scalper_hft.cli walkforward --strategy cvd_momentum --symbol BTCUSDT --interval 1m --days 60
    python -m scalper_hft.cli optimize --strategy mean_reversion --symbol BTCUSDT --interval 5m --trials 40
    python -m scalper_hft.cli overfit --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 120
    python -m scalper_hft.cli ml --symbol BTCUSDT --interval 1m --days 60
    python -m scalper_hft.cli paper --strategy mean_reversion --symbol BTCUSDT --interval 5m
    python -m scalper_hft.cli report --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 120
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from scalper_hft.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("scalper_hft.cli")


def _load_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    from scalper_hft.data.downloader import download_klines

    df = download_klines(symbol, interval, days)
    if df is None or df.empty:
        logger.error("Немає даних для %s %s — запустіть download спершу", symbol, interval)
        sys.exit(1)
    return df


def cmd_download(args: argparse.Namespace) -> None:
    from scalper_hft.data.downloader import download_agg_trades, download_funding, download_klines

    settings = get_settings()
    symbols = args.symbol.split(",") if args.symbol else list(settings.default_symbols)
    intervals = args.interval.split(",") if args.interval else ["1m"]
    for sym in symbols:
        for iv in intervals:
            df = download_klines(sym, iv, args.days, force=args.force)
            logger.info("klines %s %s: %d свічок (%s … %s)", sym, iv, len(df), df.index[0], df.index[-1])
        if args.trades:
            tr = download_agg_trades(sym, args.trades_days or min(args.days, 2), force=args.force)
            logger.info("aggTrades %s: %d трейдів", sym, len(tr) if tr is not None else 0)
        if args.funding:
            fu = download_funding(sym, args.days, force=args.force)
            logger.info("funding %s: %d точок", sym, len(fu) if fu is not None else 0)


def cmd_backtest(args: argparse.Namespace) -> None:
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy

    df = _load_klines(args.symbol, args.interval, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    trades = None
    if strategy.needs_trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)
    funding = None
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)
    res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
    print("\n" + res.summary())
    _plot_equity(res.equity, args.strategy, args.symbol)


def cmd_walkforward(args: argparse.Namespace) -> None:
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.walk_forward import run_walk_forward

    df = _load_klines(args.symbol, args.interval, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    trades = None
    if strategy.needs_trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)
    funding = None
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)
    settings = get_settings()
    res = run_walk_forward(
        df,
        strategy,
        train_bars=args.train,
        test_bars=args.test,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
    )
    print("\n" + res.summary())
    if res.avg_oos_sharpe < 0.3:
        print("\n⚠  OOS Sharpe низький — стратегія, ймовірно, не генералізує.")


def cmd_optimize(args: argparse.Namespace) -> None:
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.optimize import optimize_params

    df = _load_klines(args.symbol, args.interval, args.days)
    strategy = get_strategy(args.strategy)
    settings = get_settings()
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
    from scalper_hft.validation.optimize import optimize_ml_params

    df = _load_klines(args.symbol, args.interval, args.days)
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


def cmd_overfit(args: argparse.Namespace) -> None:
    """Повний аудит на перенавчання: WF + sensitivity + deflated Sharpe + CV."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
    from scalper_hft.validation.sensitivity import parameter_sensitivity
    from scalper_hft.validation.walk_forward import run_walk_forward

    df = _load_klines(args.symbol, args.interval, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    trades = None
    if strategy.needs_trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)
    funding = None
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)

    print("═" * 60)
    print(f"AUDIT: стратегія {args.strategy}, {args.symbol} {args.interval}, {len(df)} барів")
    print("═" * 60)

    # 1) walk-forward
    res_wf = run_walk_forward(
        df, strategy, train_bars=args.train, test_bars=args.test, trades=trades, funding=funding,
        position_pct=settings.position_pct,
    )
    print("\n[1] WALK-FORWARD")
    print(res_wf.summary())

    # 2) sensitivity головного параметра
    if strategy.param_space:
        pname = next(iter(strategy.param_space))
        lo, hi, step = strategy.param_space[pname]
        values = [lo + i * step for i in range(int((hi - lo) / step) + 1)][:15]
        res_sens = parameter_sensitivity(df, strategy, pname, values, cost=cost, trades=trades, funding=funding)
        print("\n[2] ЧУТЛИВІСТЬ ДО ПАРАМЕТРА", pname)
        print(res_sens.summary())

    # 3) deflated Sharpe на повному наборі
    res_full = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
    equity = res_full.equity
    ret = equity.pct_change().dropna()
    n_trials = estimate_n_trials(
        param_combinations=len(getattr(strategy, "param_space", {}) or {1}) or 1,
        backtests_per_combo=args.trials or 1,
    )
    dsr = deflated_sharpe_ratio(ret.values, n_trials=n_trials)
    print("\n[3] DEFLATED SHARPE (коригування на множинне тестування)")
    print(f"    raw Sharpe: {res_full.metrics.sharpe:.3f} | trials: {n_trials} | DSR: {dsr:.3f}")
    print("    DSR > 0.95 → edge статистично значущий після коригування")

    print("\n" + res_full.summary())


def cmd_ml(args: argparse.Namespace) -> None:
    """Walk-forward ML: Triple-Barrier + LightGBM + AFML sample weights."""
    from scalper_hft.ml.trainer import train_from_ohlcv

    df = _load_klines(args.symbol, args.interval, args.days)
    trades = None
    if args.trades:
        from scalper_hft.data.downloader import download_agg_trades
        trades = download_agg_trades(args.symbol, args.days)

    mode = getattr(args, "mode", "triple_barrier")
    pt   = float(getattr(args, "pt", 1.0))
    sl   = float(getattr(args, "sl", 1.0))
    holding = int(getattr(args, "holding", 10))
    decay    = float(getattr(args, "decay", 0.9))
    frac_d   = float(getattr(args, "frac_d", 0.4))
    no_frac  = bool(getattr(args, "no_frac_diff", False))

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
        )
    except ValueError as e:
        logger.error("ML тренування: %s", e)
        sys.exit(1)

    print("\n" + res.summary())

    if res.feature_importance is not None:
        print("\nТоп-10 фіч (gain):")
        print(res.feature_importance.head(10).to_string())


def cmd_paper(args: argparse.Namespace) -> None:
    from scalper_hft.live.trader import LiveTrader, run_trader_once
    from scalper_hft.strategies import get_strategy

    df = _load_klines(args.symbol, args.interval, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    trader = LiveTrader(strategy, args.symbol, args.interval)
    result = run_trader_once(trader, df)
    print(f"Останній бар: {df.index[-1]}")
    print(f"Дія: {result}")
    print(f"Капітал: {trader.account.equity:.2f} | позиції: {len(trader.account.positions)} | угод: {len(trader.account.trades)}")


def cmd_report(args: argparse.Namespace) -> None:
    """Markdown-звіт: бектест + WF + sensitivity + deflated Sharpe → docs/reports/."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
    from scalper_hft.validation.sensitivity import parameter_sensitivity
    from scalper_hft.validation.walk_forward import run_walk_forward

    df = _load_klines(args.symbol, args.interval, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    trades = None
    if strategy.needs_trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)
    funding = None
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)

    res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct)
    wf = run_walk_forward(
        df, strategy, train_bars=args.train, test_bars=args.test, trades=trades, funding=funding,
        position_pct=settings.position_pct,
    )
    ret = res.equity.pct_change().dropna()
    n_trials = estimate_n_trials(max(len(strategy.param_space), 1), args.trials or 1)
    dsr = deflated_sharpe_ratio(ret.values, n_trials=n_trials)

    sens_md = ""
    if strategy.param_space:
        pname = next(iter(strategy.param_space))
        lo, hi, step = strategy.param_space[pname]
        values = [lo + i * step for i in range(int((hi - lo) / step) + 1)][:15]
        try:
            sens = parameter_sensitivity(df, strategy, pname, values, cost=cost, trades=trades, funding=funding)
            sens_md = f"\n## Sensitivity ({pname})\n\nsmoothness = {sens.smoothness:.3f}\n\n" + sens.grid.to_markdown(index=False)
        except Exception as exc:  # noqa: BLE001
            sens_md = f"\n## Sensitivity\n\nпомилка: {exc}"

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

## Deflated Sharpe

- raw Sharpe: {res.metrics.sharpe:.3f}
- trials: {n_trials}
- **DSR: {dsr:.3f}** {'✅ edge значущий' if dsr > 0.95 else '⚠ edge не підтверджено'}
{sens_md}

## Висновок

- OOS Sharpe: {wf.avg_oos_sharpe:.3f} ({wf.positive_windows_frac:.0%} вікон > 0)
- DSR: {dsr:.3f}
- {'Стратегія готова до paper trading' if wf.avg_oos_sharpe > 0.3 and dsr > 0.9 else 'Стратегія потребує доопрацювання'}
"""
    out_dir = Path("docs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.strategy}_{args.symbol}_{args.interval}.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nЗвіт збережено: {out_path}")


def cmd_cscv(args: argparse.Namespace) -> None:
    """PBO через Combinatorial Purged CV (López de Prado)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.cscv import pbo_cscv, variant_returns

    df = _load_klines(args.symbol, args.interval, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    trades = None
    if strategy.needs_trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)
    funding = None
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days)

    print(f"Генерація {args.variants} випадкових варіантів параметрів {args.strategy}...")
    returns = variant_returns(
        df, strategy, n_variants=args.variants, cost=cost, trades=trades, funding=funding,
        position_pct=settings.position_pct,
    )
    res = pbo_cscv(returns, n_blocks=args.blocks, threshold=0.0, max_combos=args.max_combos)
    print("\n" + res.summary())


def cmd_record_bookticker(args: argparse.Namespace) -> None:
    """Запис bookTicker у реальному часі (для OB-стратегій)."""
    from scalper_hft.live.bookticker_recorder import record_bookticker, record_depth

    for sym in (args.symbol or "BTCUSDT").split(","):
        if args.depth:
            n = record_depth(sym, minutes=args.minutes)
            print(f"{sym}: записано {n} depth5 снапшотів")
        else:
            n = record_bookticker(sym, minutes=args.minutes)
            print(f"{sym}: записано {n} bookTicker снапшотів")


def cmd_paper_run(args: argparse.Namespace) -> None:
    """Циклічний paper-прогін: кілька кроків з паузою, збереження угод."""
    from scalper_hft.live.paper_runner import PaperRunner
    from scalper_hft.strategies import get_strategy

    strategy = get_strategy(args.strategy, **args.param_dict)
    runner = PaperRunner(strategy, args.symbol, args.interval)
    result = runner.run(iterations=args.iterations, sleep_sec=args.sleep)
    print("\n" + result.summary())
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        send_telegram(f"Paper-run {args.strategy} {args.symbol}: {result.actions[-1]}, equity={result.account.equity:.2f}")


def cmd_paper_replay(args: argparse.Namespace) -> None:
    """Відтворення історії через риск-контрольованого трейдера (валідація risk-шару)."""
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.live.paper_replay import paper_replay
    from scalper_hft.strategies import get_strategy

    df = _load_klines(args.symbol, args.interval, args.days)
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


def cmd_arb(args: argparse.Namespace) -> None:
    """Delta-neutral funding arbitrage (перп+спот): бектест + walk-forward."""
    from scalper_hft.backtest.delta_neutral import run_delta_neutral_backtest, run_dn_walk_forward
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.downloader import download_funding, download_klines, download_spot_klines
    from scalper_hft.strategies import get_strategy

    perp = download_klines(args.symbol, args.interval, args.days)
    spot = download_spot_klines(args.symbol, args.interval, args.days)
    funding = download_funding(args.symbol, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)

    res = run_delta_neutral_backtest(
        perp, spot, strategy, funding, position_pct=args.position_pct or 0.1, cost=cost,
        maker_execution=args.maker,
    )
    print("\n" + res.summary())
    _plot_equity(res.equity, args.strategy, args.symbol)

    if args.walkforward:
        wf = run_dn_walk_forward(
            perp, spot, strategy, funding, train_bars=args.train, test_bars=args.test,
            position_pct=args.position_pct or 0.1,
        )
        print(
            f"\nWalk-forward: {wf['n_windows']} вікон | avg IS SRh={wf['avg_is_sharpe']:+.3f} | "
            f"avg OOS SRh={wf['avg_oos_sharpe']:+.3f} | позитивних OOS: {wf['positive_windows']:.0%}"
        )
        if wf["avg_oos_sharpe"] < 0.1:
            print("⚠  OOS слабкий — edge не підтверджено")


def cmd_pairs(args: argparse.Namespace) -> None:
    """Статистичний арбітраж пар (BTC/ETH/SOL перпи)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs import run_pairs_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.downloader import download_funding, download_klines
    from scalper_hft.strategies import get_strategy

    leg1, leg2 = (args.leg1 or "BTCUSDT"), (args.leg2 or "ETHUSDT")
    df1 = download_klines(leg1, args.interval, args.days)
    df2 = download_klines(leg2, args.interval, args.days)
    f1 = download_funding(leg1, args.days)
    f2 = download_funding(leg2, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)

    res = run_pairs_backtest(
        df1, df2, strategy, f1, f2,
        position_pct=args.position_pct or 0.1, cost=cost, maker_execution=args.maker,
    )
    print(f"\nПара: {leg1} / {leg2} ({args.interval}, {len(res.spread)} спільних барів)")
    print(res.summary())
    _plot_equity(res.equity, args.strategy, f"{leg1}_{leg2}")

    if args.walkforward:
        from scalper_hft.backtest.pairs import run_pairs_walk_forward

        wf = run_pairs_walk_forward(
            df1, df2, strategy, f1, f2,
            train_bars=args.train, test_bars=args.test,
            position_pct=args.position_pct or 0.1, maker_execution=args.maker,
        )
        print(
            f"\nWalk-forward: {wf['n_windows']} вікон | IS SRh={wf['avg_is_sharpe']:+.3f} | "
            f"OOS SRh={wf['avg_oos_sharpe']:+.3f} | позитивних OOS: {wf['positive_windows']:.0%}"
        )


def cmd_pairs_portfolio(args: argparse.Namespace) -> None:
    """Бектест портфеля валідованих пар (XRP/BTC + BTC/ETH + LINK/BTC)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs_portfolio import run_pairs_portfolio
    from scalper_hft.config import get_settings
    from scalper_hft.data.downloader import download_funding, download_klines
    from scalper_hft.live.pairs_runner import VALIDATED_PAIRS
    from scalper_hft.strategies.pairs_arb import PairsArb

    interval = args.interval or "1h"
    settings = get_settings()
    symbols = sorted({c["leg1"] for c in VALIDATED_PAIRS} | {c["leg2"] for c in VALIDATED_PAIRS})
    data = {sym: download_klines(sym, interval, args.days) for sym in symbols}
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
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    res = run_pairs_portfolio(
        data, configs, position_pct=args.position_pct or 0.3, cost=cost, maker_execution=True
    )
    print(res.summary())
    _plot_equity(res.equity, "pairs_portfolio", "validated")


def cmd_paper_run_pairs(args: argparse.Namespace) -> None:
    """Циклічний paper pairs (maker, 2 ноги) або портфель валідованих пар."""
    from scalper_hft.live.pairs_runner import PairsPaperRunner, PairsPortfolioRunner
    from scalper_hft.live.store import PaperStore
    from scalper_hft.strategies import get_strategy

    store = PaperStore()
    interval = args.interval or "1h"
    if args.portfolio:
        runner = PairsPortfolioRunner(interval=interval, store=store, is_maker=True)
    else:
        strategy = get_strategy(args.strategy or "pairs_arb", **args.param_dict)
        runner = PairsPaperRunner(
            args.leg1 or "XRPUSDT",
            args.leg2 or "BTCUSDT",
            interval=interval,
            strategy=strategy,
            store=store,
            is_maker=True,
        )
    result = runner.run(iterations=args.iterations, sleep_sec=args.sleep)
    print("\n" + result.summary())
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        send_telegram(
            f"Paper pairs {result.pair}: {result.actions[-1] if result.actions else '-'} | "
            f"equity={result.account.equity:.2f} fill={result.n_filled}/{result.n_filled + result.n_unfilled}"
        )


def cmd_paper_replay_pairs(args: argparse.Namespace) -> None:
    """Історичний paper pairs з моделлю unfilled post-only."""
    from scalper_hft.data.downloader import download_funding, download_klines
    from scalper_hft.live.pairs_runner import replay_pairs
    from scalper_hft.live.store import PaperStore
    from scalper_hft.strategies import get_strategy

    interval = args.interval or "1h"
    leg1, leg2 = (args.leg1 or "XRPUSDT"), (args.leg2 or "BTCUSDT")
    df1 = download_klines(leg1, interval, args.days)
    df2 = download_klines(leg2, interval, args.days)
    strategy = get_strategy(args.strategy or "pairs_arb", **args.param_dict)
    store = PaperStore()
    result = replay_pairs(
        leg1, leg2, df1, df2, strategy=strategy, store=store,
        funding1=download_funding(leg1, args.days),
        funding2=download_funding(leg2, args.days),
        is_maker=True, interval=interval,
    )
    print("\n" + result.summary())
    _plot_equity(result.equity, "paper_pairs", f"{leg1}_{leg2}")
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        send_telegram(result.summary())


def _plot_equity(equity: pd.Series, strategy: str, symbol: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        equity.plot(ax=ax, title=f"{strategy} · {symbol}")
        ax.set_ylabel("Equity")
        out = Path("docs/plots")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{strategy}_{symbol}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        logger.info("Графік збережено: %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалося побудувати графік: %s", exc)


def _parse_param_dict(args: list[str]) -> dict:
    out: dict = {}
    for item in args or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scalper-hft", description="Високочастотна скальпінг-система (Binance USDT-M)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbol", default=None, help="Символ, e.g. BTCUSDT (за замовч. з .env)")
        p.add_argument("--interval", default=None, help="Таймфрейм: 1s/5s/1m/5m/15m/1h")
        p.add_argument("--strategy", default="mean_reversion", help="Стратегія з реєстру: mean_reversion, cvd_momentum, pairs_arb, ...")
        p.add_argument("--days", type=int, default=60, help="Глибина історії, днів")
        p.add_argument("-p", "--param", action="append", default=[], help="Параметр стратегії: key=value")

    p = sub.add_parser("download", help="Завантажити klines/aggTrades/funding")
    add_common(p)
    p.add_argument("--trades", action="store_true", help="Також aggTrades")
    p.add_argument("--trades-days", type=int, default=None, help="Глибина aggTrades (Binance обмежує 2 доби)")
    p.add_argument("--funding", action="store_true", help="Також funding")
    p.add_argument("--force", action="store_true", help="Ігнорувати кеш")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("backtest", help="Запустити бектест")
    add_common(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("walkforward", help="Walk-forward аналіз")
    add_common(p)
    p.add_argument("--train", type=int, default=2000, help="Барів у train (IS)")
    p.add_argument("--test", type=int, default=500, help="Барів у test (OOS)")
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("optimize", help="Оптимізація параметрів (Optuna)")
    add_common(p)
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--splits", type=int, default=4)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("ml-opt", help="Оптимізація параметрів ML-моделі (AFML)")
    add_common(p)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--scoring", default="neg_log_loss")
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--frac-d", type=float, default=0.4)
    p.add_argument("--no-frac-diff", action="store_true")
    p.add_argument("--trades", action="store_true", help="Використовувати aggTrades")
    p.add_argument("--sampler", default="tpe", choices=["tpe", "random"])
    p.set_defaults(func=cmd_ml_opt)

    p = sub.add_parser("overfit", help="Повний аудит на перенавчання")
    add_common(p)
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--trials", type=int, default=50, help="Оцінка кількості спроб для DSR")
    p.set_defaults(func=cmd_overfit)

    p = sub.add_parser("cscv", help="PBO через Combinatorial Purged CV")
    add_common(p)
    p.add_argument("--variants", type=int, default=30, help="Кількість випадкових варіантів параметрів")
    p.add_argument("--blocks", type=int, default=8, help="Кількість блоків для розбиття")
    p.add_argument("--max-combos", type=int, default=200, help="Обмеження комбінацій")
    p.set_defaults(func=cmd_cscv)

    p = sub.add_parser("record-bookticker", help="Запис bookTicker/depth5 (WS) у parquet")
    p.add_argument("--symbol", default="BTCUSDT", help="Символ(и) через кому")
    p.add_argument("--minutes", type=int, default=60, help="Тривалість запису, хв")
    p.add_argument("--depth", action="store_true", help="Записувати depth5 (5 рівнів стакана) замість bookTicker")
    p.set_defaults(func=cmd_record_bookticker)

    p = sub.add_parser("arb", help="Delta-neutral funding arb (перп+спот)")
    add_common(p)
    p.add_argument("--position-pct", type=float, default=None, help="Ноціонал кожної ноги (за замовч. 0.1)")
    p.add_argument("--maker", action="store_true", help="Комісії maker (post-only) на обох ногах")
    p.add_argument("--walkforward", action="store_true", help="Додатково walk-forward")
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--test", type=int, default=5000)
    p.set_defaults(func=cmd_arb)

    p = sub.add_parser("pairs", help="Статистичний арбітраж пар перпів (BTC/ETH/SOL)")
    add_common(p)
    p.add_argument("--leg1", default="BTCUSDT", help="Перша нога")
    p.add_argument("--leg2", default="ETHUSDT", help="Друга нога")
    p.add_argument("--position-pct", type=float, default=None, help="Ноціонал кожної ноги")
    p.add_argument("--maker", action="store_true", help="Комісії maker (post-only)")
    p.add_argument("--walkforward", action="store_true", help="Додатково walk-forward")
    p.add_argument("--train", type=int, default=1500)
    p.add_argument("--test", type=int, default=500)
    p.set_defaults(func=cmd_pairs)

    p = sub.add_parser("pairs-portfolio", help="Бектест портфеля валідованих пар")
    add_common(p)
    p.add_argument("--position-pct", type=float, default=0.3)
    p.set_defaults(func=cmd_pairs_portfolio, interval="1h")

    p = sub.add_parser("paper-run-pairs", help="Paper pairs (maker, 2 ноги) або --portfolio")
    add_common(p)
    p.add_argument("--leg1", default="XRPUSDT")
    p.add_argument("--leg2", default="BTCUSDT")
    p.add_argument("--portfolio", action="store_true", help="Три валідовані пари на спільному рахунку")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--sleep", type=int, default=300, help="Пауза між кроками, сек (для 1h — 300+)")
    p.add_argument("--notify", action="store_true")
    p.set_defaults(func=cmd_paper_run_pairs, strategy="pairs_arb", interval="1h")

    p = sub.add_parser("paper-replay-pairs", help="Історичний paper pairs з моделлю unfilled")
    add_common(p)
    p.add_argument("--leg1", default="XRPUSDT")
    p.add_argument("--leg2", default="BTCUSDT")
    p.add_argument("--notify", action="store_true")
    p.set_defaults(func=cmd_paper_replay_pairs, strategy="pairs_arb", interval="1h")

    p = sub.add_parser("ml", help="Walk-forward ML-класифікатор: Triple-Barrier + LightGBM + AFML")
    add_common(p)
    p.add_argument("--mode", default="triple_barrier", choices=["triple_barrier", "horizon"],
                   help="Режим лейблінгу: triple_barrier (AFML, default) або horizon")
    p.add_argument("--pt", type=float, default=1.0, help="Profit-take множник (× ATR)")
    p.add_argument("--sl", type=float, default=1.0, help="Stop-loss множник (× ATR)")
    p.add_argument("--holding", type=int, default=10, help="Вертикальний бар'єр (барів)")
    p.add_argument("--decay", type=float, default=0.9, help="Time-decay для sample weights")
    p.add_argument("--frac-d", type=float, default=0.4, dest="frac_d",
                   help="Ступінь fractional differencing")
    p.add_argument("--no-frac-diff", action="store_true", dest="no_frac_diff",
                   help="Вимкнути frac_diff фічі")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--trades", action="store_true", help="Використати aggTrades (CVD фічі)")
    p.set_defaults(func=cmd_ml)

    p = sub.add_parser("paper", help="Один крок paper trading на останньому барі")
    add_common(p)
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("paper-run", help="Циклічний paper-прогін (кілька кроків)")
    add_common(p)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--sleep", type=int, default=60, help="Пауза між кроками, сек")
    p.add_argument("--notify", action="store_true", help="Telegram-сповіщення після прогіну")
    p.set_defaults(func=cmd_paper_run)

    p = sub.add_parser("paper-replay", help="Відтворення історії через risk-трейдера")
    add_common(p)
    p.add_argument("--position-pct", type=float, default=None, help="Частка капіталу на позицію")
    p.add_argument("--notify", action="store_true", help="Telegram-сповіщення результату")
    p.set_defaults(func=cmd_paper_replay)

    p = sub.add_parser("report", help="Markdown-звіт у docs/reports/")
    add_common(p)
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--trials", type=int, default=50)
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.param_dict = _parse_param_dict(getattr(args, "param", []))
    args.func(args)


if __name__ == "__main__":
    main()
