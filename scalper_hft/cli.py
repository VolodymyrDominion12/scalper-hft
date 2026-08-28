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
    from scalper_hft.data.downloader import download_agg_trades
    from scalper_hft.ml.features import build_labeled_dataset
    from scalper_hft.ml.trainer import train_walk_forward

    df = _load_klines(args.symbol, args.interval, args.days)
    trades = download_agg_trades(args.symbol, args.days) if args.trades else None
    X, y = build_labeled_dataset(df, horizon=args.horizon, trades=trades)
    if len(X) < 1000:
        logger.error("Замало labeled-прикладів (%d) — збільшіть days", len(X))
        sys.exit(1)
    res = train_walk_forward(X, y, train_size=args.train, test_size=args.test, close=df["close"])
    print("\n" + res.summary())


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
        p.add_argument("--strategy", default="mean_reversion", help="Стратегія з реєстру: mean_reversion, cvd_momentum, ob_imbalance, market_maker")
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

    p = sub.add_parser("ml", help="Walk-forward ML-класифікатор напрямку")
    add_common(p)
    p.add_argument("--horizon", type=int, default=3, help="Горизонт прогнозу, барів")
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
