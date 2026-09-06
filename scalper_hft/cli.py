"""CLI для scalper-hft.

Приклади:
    python -m scalper_hft.cli download --symbol BTCUSDT --interval 1m --days 30
    python -m scalper_hft.cli backtest --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 90
    python -m scalper_hft.cli plot --strategy mean_reversion --symbol BTCUSDT --interval 5m --days 30
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

_SHORT_PAIRS_INTERVALS = frozenset({"1m", "5m"})


def _warn_pairs_short_interval(strategy: str, interval: str | None) -> None:
    if strategy == "pairs_arb" and str(interval) in _SHORT_PAIRS_INTERVALS:
        logger.warning(
            "pairs_arb на %s: fee-drag уже відхилив 1m; валідований edge — 1h maker",
            interval,
        )


def _apply_use_kalman(args: argparse.Namespace, params: dict) -> dict:
    out = dict(params)
    if getattr(args, "use_kalman", False):
        out["use_kalman"] = True
    return out


def _load_klines(
    symbol: str, interval: str, days: int, base: str | None = None, derive: bool = True, exchange_id: str | None = None
) -> pd.DataFrame:
    """Хелпер для CLI: завантажити/ресемплити дані або впасти з помилкою."""
    from scalper_hft.data.access import ensure_klines

    df = ensure_klines(symbol, interval, days, base_interval=base or "1m", derive=derive, exchange_id=exchange_id)
    if df is None or df.empty:
        logger.error("Немає даних %s %s — запустіть download спершу", symbol, interval)
        sys.exit(1)
    return df


def cmd_download(args: argparse.Namespace) -> None:
    from scalper_hft.data.downloader import download_agg_trades, download_funding, download_klines

    settings = get_settings()
    symbols = args.symbol.split(",") if args.symbol else list(settings.default_symbols)
    intervals = args.interval.split(",") if args.interval else ["1m"]
    retries = getattr(args, "retries", None)
    batch_delay = getattr(args, "delay", None)
    checkpoint_batches = getattr(args, "checkpoint_batches", None)
    exchange_id = getattr(args, "exchange", settings.exchange)

    logger.info("Спочатку звірю кеш: докачаю лише відсутні дні/вікна (--force оновлює хвіст)")
    for sym in symbols:
        for iv in intervals:
            df = download_klines(
                sym,
                iv,
                args.days,
                force=args.force,
                retries=retries,
                batch_delay=batch_delay,
                checkpoint_batches=checkpoint_batches,
                exchange_id=exchange_id,
            )
            logger.info("klines %s %s: %d свічок (%s … %s)", sym, iv, len(df), df.index[0], df.index[-1])
        if args.trades:
            tr = download_agg_trades(
                sym,
                args.trades_days or min(args.days, 2),
                force=args.force,
                retries=retries,
                batch_delay=batch_delay,
                checkpoint_batches=checkpoint_batches,
                exchange_id=exchange_id,
            )
            logger.info("aggTrades %s: %d трейдів", sym, len(tr) if tr is not None else 0)
        if args.funding:
            fu = download_funding(
                sym,
                args.days,
                force=args.force,
                retries=retries,
                batch_delay=batch_delay,
                checkpoint_batches=checkpoint_batches,
                exchange_id=exchange_id,
            )
            logger.info("funding %s: %d точок", sym, len(fu) if fu is not None else 0)
        if getattr(args, "vision", False):
            from datetime import date, timedelta

            from scalper_hft.data.binance_vision import download_agg_trades_vision

            start = (
                date.fromisoformat(args.vision_start) if args.vision_start else date.today() - timedelta(days=args.days)
            )
            tr = download_agg_trades_vision(sym, start=start, freq=args.vision_freq)
            logger.info("vision aggTrades %s: %d трейдів", sym, len(tr) if tr is not None else 0)


def cmd_backtest(args: argparse.Namespace) -> None:
    if getattr(args, "enqueue", False):
        from scalper_hft.research.job_handlers import payload_from_backtest_cli

        _enqueue_job("backtest", payload_from_backtest_cli(args))
        return
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.research import load_research_data
    from scalper_hft.strategies import get_strategy

    params = _apply_use_kalman(args, dict(args.param_dict))
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
            logger.error("Немає даних aggTrades для формування барів")
            sys.exit(1)
        threshold = getattr(args, "bar_threshold", 100000.0)
        df = create_dollar_bars(trades, threshold) if bar_type == "dollar" else create_volume_bars(trades, threshold)
        logger.info("Згенеровано %d %s-барів", len(df), bar_type)
    if bundle.quality is not None and not bundle.quality.ok:
        logger.warning("Якість барів: %s", bundle.quality.summary())
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    res = run_strategy_backtest(
        df, strategy, cost=cost, trades=trades, funding=funding, position_pct=settings.position_pct
    )
    print("\n" + res.summary())
    _plot_equity(res.equity, args.strategy, args.symbol)


def cmd_plot(args: argparse.Namespace) -> None:
    """Інтерактивний HTML-графік бектесту: свічки + індикатори + угоди + SL/TP.

    Зберігає standalone HTML (Plotly) у --out — відкривається у будь-якому
    браузері без сервера: зум, hover, легенда-перемикачі.
    """
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.research import load_research_data
    from scalper_hft.features.indicators import add_standard_features
    from scalper_hft.strategies import get_strategy
    from scalper_hft.visualization.charts import make_backtest_figure

    strategy = get_strategy(args.strategy, **dict(args.param_dict))
    settings = get_settings()
    bundle = load_research_data(
        args.symbol,
        args.interval,
        args.days,
        strategy,
        base=getattr(args, "base", None) or "1m",
        derive=getattr(args, "derive", True),
        exchange_id=getattr(args, "exchange", settings.exchange),
    )
    df = bundle.klines
    if df is None or df.empty:
        logger.error("Немає даних %s %s — запустіть download спершу", args.symbol, args.interval)
        sys.exit(1)
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    res = run_strategy_backtest(
        df, strategy, cost=cost, trades=bundle.trades, funding=bundle.funding, position_pct=settings.position_pct
    )
    fdf = add_standard_features(df)  # індикатори — лише для графіка
    fig = make_backtest_figure(
        fdf,
        res,
        symbol=args.symbol,
        max_bars=args.bars,
        start=args.start,
        end=args.end,
        with_trades=not args.no_trades,
        with_sl_tp=not args.no_sl_tp,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    logger.info("Графік збережено: %s (%d угод)", out, len(res.trades))


def cmd_walkforward(args: argparse.Namespace) -> None:
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.walk_forward import run_walk_forward

    settings = get_settings()
    exchange_id = getattr(args, "exchange", settings.exchange)
    df = _load_klines(
        args.symbol,
        args.interval,
        args.days,
        base=getattr(args, "base", None),
        derive=getattr(args, "derive", True),
        exchange_id=exchange_id,
    )
    strategy = get_strategy(args.strategy, **_apply_use_kalman(args, args.param_dict))
    trades = None
    if strategy.needs_trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days, exchange_id=exchange_id)
    funding = None
    if strategy.needs_funding:
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(args.symbol, args.days, exchange_id=exchange_id)
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

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
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


def _param_combinations(strategy) -> int:
    """Кількість комбінацій параметрів у param_space (добуток розмірів ґраток).

    Раніше в DSR передавався `len(param_space)` (кількість ПАРАМЕТРІВ, напр. 3),
    а не кількість спроб/комбінацій → корекція на множинне тестування була
    занижена в рази. Тут — добуток кількості значень по кожному параметру
    (обрізаний зверху, щоб не вибухало).
    """
    ps = getattr(strategy, "param_space", {}) or {}
    if not ps:
        return 1
    combos = 1
    for lo, hi, step in ps.values():
        step = float(step) if step else 1.0
        n_vals = max(int((float(hi) - float(lo)) / step) + 1, 1)
        combos *= n_vals
    return min(max(combos, 1), 100_000)


def cmd_overfit(args: argparse.Namespace) -> None:
    """Повний аудит на перенавчання: WF + sensitivity + deflated Sharpe + CV."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
    from scalper_hft.validation.sensitivity import parameter_sensitivity
    from scalper_hft.validation.walk_forward import run_walk_forward

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    strategy = get_strategy(args.strategy, **_apply_use_kalman(args, args.param_dict))
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
        df,
        strategy,
        train_bars=args.train,
        test_bars=args.test,
        trades=trades,
        funding=funding,
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
        param_combinations=_param_combinations(strategy),
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
        logger.error("ML тренування: %s", e)
        sys.exit(1)

    print("\n" + res.summary())

    if res.feature_importance is not None:
        print("\nТоп-10 фіч (gain):")
        print(res.feature_importance.head(10).to_string())


def cmd_paper(args: argparse.Namespace) -> None:
    from scalper_hft.live.trader import LiveTrader, run_trader_once
    from scalper_hft.strategies import get_strategy

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


def cmd_report(args: argparse.Namespace) -> None:
    """Markdown-звіт: бектест + WF + sensitivity + deflated Sharpe → docs/reports/."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio, estimate_n_trials
    from scalper_hft.validation.sensitivity import parameter_sensitivity
    from scalper_hft.validation.walk_forward import run_walk_forward

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
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
        df,
        strategy,
        train_bars=args.train,
        test_bars=args.test,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
    )
    ret = res.equity.pct_change().dropna()
    n_trials = estimate_n_trials(_param_combinations(strategy), args.trials or 1)
    dsr = deflated_sharpe_ratio(ret.values, n_trials=n_trials)

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
    quintile_md = ""
    try:
        from scalper_hft.validation.quintile import quintile_spread_study

        signals = strategy.generate_signals(df, trades=trades, funding=funding)
        fwd_ret = df["close"].pct_change().shift(-1).fillna(0.0)
        if signals.abs().sum() > 5:
            q_res = quintile_spread_study(signals.astype(float), fwd_ret)
            quintile_md = f"\n## Quintile Study (монотонність сигналу)\n\n```\n{q_res.summary()}\n```\n"
    except Exception as exc:  # noqa: BLE001
        quintile_md = f"\n## Quintile Study\n\nпомилка: {exc}\n"

    # ── Time-decay test (Narang гл. 9) ───────────────────────────────────────
    decay_md = ""
    try:
        from scalper_hft.validation.time_decay import time_decay_test

        td_res = time_decay_test(df, strategy, max_lag=3, cost=cost, trades=trades, funding=funding)
        decay_md = f"\n## Time-Decay Test (лаг входу)\n\n```\n{td_res.summary()}\n```\n"
        if len(td_res.sharpes) >= 2 and td_res.sharpes[0] > 0 and td_res.sharpes[1] < td_res.sharpes[0] * 0.5:
            decay_md += "\n> ⚠ Альфа різко втрачається при лазі 1 — бектест може переоцінювати edge!\n"
    except Exception as exc:  # noqa: BLE001
        decay_md = f"\n## Time-Decay Test\n\nпомилка: {exc}\n"

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
- **DSR: {dsr:.3f}** {"✅ edge значущий" if dsr > 0.95 else "⚠ edge не підтверджено"}
{sens_md}{quintile_md}{decay_md}
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
    print(f"\nЗвіт збережено: {out_path}")


def cmd_cscv(args: argparse.Namespace) -> None:
    """PBO через Combinatorial Purged CV (López de Prado)."""
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.cscv import pbo_cscv, variant_returns

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
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
        df,
        strategy,
        n_variants=args.variants,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
    )
    res = pbo_cscv(returns, n_blocks=args.blocks, threshold=0.0, max_combos=args.max_combos)
    print("\n" + res.summary())


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
    from scalper_hft.config import get_settings
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
        slippage=settings.slippage,
    )

    strat_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not strat_names:
        logger.error("--strategies не вказано або порожнє")
        return

    # ── Бектест базових стратегій ────────────────────────────────────────
    from scalper_hft.backtest.engine import BacktestEngine

    baseline_returns: dict[str, pd.Series] = {}
    for name in strat_names:
        try:
            strat = get_strategy(name)
            engine = BacktestEngine(df, strat, cost_model=cost)
            result = engine.run()
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
            engine = BacktestEngine(df, sup, cost_model=cost)
            result = engine.run()
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


def cmd_telegram_bot(args: argparse.Namespace) -> None:
    """Запустити інтерактивний Telegram Bot."""
    from scalper_hft.live.telegram_bot import TelegramBotServer

    store_path = getattr(args, "store", None) or Path("results") / "paper_pairs.sqlite"
    control_path = getattr(args, "control", None) or Path("results") / "control.json"

    server = TelegramBotServer(
        store_path=Path(store_path),
        control_path=Path(control_path),
    )
    logger.info("Telegram Bot стартує (Ctrl+C для зупинки)")
    try:
        server.run_polling()
    except KeyboardInterrupt:
        server.stop()
        logger.info("Telegram Bot зупинено")


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

        eq_str = f"{result.account.equity:.2f}" if result.account else "N/A"
        send_telegram(f"Paper-run {args.strategy} {args.symbol}: {result.actions[-1]}, equity={eq_str}")


def cmd_paper_replay(args: argparse.Namespace) -> None:
    """Відтворення історії через риск-контрольованого трейдера (валідація risk-шару)."""
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


def cmd_arb(args: argparse.Namespace) -> None:
    """Delta-neutral funding arbitrage (перп+спот): бектест + walk-forward."""
    from scalper_hft.backtest.delta_neutral import run_delta_neutral_backtest, run_dn_walk_forward
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.data.downloader import download_funding, download_spot_klines
    from scalper_hft.strategies import get_strategy

    perp = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
    spot = download_spot_klines(args.symbol, args.interval, args.days)
    funding = download_funding(args.symbol, args.days)
    strategy = get_strategy(args.strategy, **args.param_dict)
    settings = get_settings()
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)

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


def cmd_pairs(args: argparse.Namespace) -> None:
    """Статистичний арбітраж пар (BTC/ETH/SOL перпи)."""
    if getattr(args, "enqueue", False):
        from scalper_hft.research.job_handlers import payload_from_pairs_cli

        _enqueue_job("pairs", payload_from_pairs_cli(args))
        return
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs import run_pairs_backtest
    from scalper_hft.config import get_settings
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
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)

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
    from scalper_hft.config import get_settings
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
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
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


def cmd_paper_run_pairs(args: argparse.Namespace) -> None:
    """Циклічний paper pairs (maker, 2 ноги) або портфель валідованих пар."""
    from pathlib import Path

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
    from pathlib import Path

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
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.cohort import cohort_report

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
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
    print(f"\nCohort: {args.strategy} · {args.symbol} · {args.interval} · {len(res.trades)} угод\n")
    print(cohort_report(res.trades, freq=args.freq))


def cmd_lift(args: argparse.Namespace) -> None:
    """Децильний lift-аналіз фіч (uplift-концепт, Predictive Marketing Ch.2/9)."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
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
    trades = None
    if strategy.needs_trades:
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(args.symbol, args.days)
    funding = None
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
        logger.error("featimp: %s", e)
        sys.exit(1)

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
        logger.error("cfi: %s", e)
        sys.exit(1)

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
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.stress import SCENARIOS, stress_report

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
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
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.capacity import capacity_report

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
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
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.survival import kaplan_meier, median_survival_time, trade_durations

    df = _load_klines(
        args.symbol, args.interval, args.days, base=getattr(args, "base", None), derive=getattr(args, "derive", True)
    )
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


def cmd_migrate_to_parquet(args: argparse.Namespace) -> None:
    """Міграція всіх даних з PostgreSQL у Parquet-файли (data/).

    Читає klines / aggTrades / funding з PostgresStore і зберігає їх
    у ParquetStore без змін формату. Ідемпотентно: вже наявні файли
    пропускаються (без --overwrite). Після успішного завершення треба
    змінити DATA_BACKEND=parquet у .env.
    """
    import importlib.util
    from pathlib import Path as _Path

    script = _Path(__file__).resolve().parent.parent / "scripts" / "migrate_postgres_to_parquet.py"
    spec = importlib.util.spec_from_file_location("migrate_pg", script)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    symbols = (
        [s.strip().upper() for s in args.symbol.split(",") if s.strip()] if getattr(args, "symbol", None) else None
    )
    data_dir = _Path(args.data_dir) if getattr(args, "data_dir", None) else None

    mod.migrate(
        symbols=symbols,
        overwrite=getattr(args, "overwrite", False),
        skip_trades=getattr(args, "skip_trades", False),
        skip_funding=getattr(args, "skip_funding", False),
        data_dir=data_dir,
        dry_run=getattr(args, "dry_run", False),
    )


def cmd_mcp(args: argparse.Namespace) -> None:
    """Запуск MCP-сервера для трейдінгу (stdio, JSON-RPC)."""
    from scalper_hft.mcp_trading import run_stdio

    run_stdio()


def cmd_dashboard_hash(args: argparse.Namespace) -> None:
    """Генерація хешу паролю для DASHBOARD_PASSWORD_HASH."""
    import getpass

    from scalper_hft.dashboard_auth import generate_hash

    pwd1 = getpass.getpass("Введіть новий пароль для дашборду: ")
    pwd2 = getpass.getpass("Повторіть пароль: ")
    if pwd1 != pwd2:
        print("❌ Паролі не співпадають.")
        sys.exit(1)

    if not pwd1:
        print("❌ Пароль не може бути порожнім.")
        sys.exit(1)

    hash_str = generate_hash(pwd1)
    print("\n✅ Пароль успішно захешовано!")
    print("Додайте цей рядок до вашого файлу .env:\n")
    print(f"DASHBOARD_PASSWORD_HASH='{hash_str}'")
    print("\nПісля цього перезапустіть дашборд.")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Запуск Streamlit-дашборду тим самим Python, що й CLI (не Anaconda PATH)."""
    import subprocess

    script = Path(__file__).resolve().parent / "dashboard.py"
    try:
        import streamlit  # noqa: F401
    except ImportError:
        logger.error("Немає streamlit. Встановіть: uv pip install -e '.[dashboard]'")
        sys.exit(1)
    cmd = [sys.executable, "-m", "streamlit", "run", str(script)]
    if args.port is not None:
        cmd.extend(["--server.port", str(args.port)])
    worker_proc = None
    if not getattr(args, "no_worker", False):
        from scalper_hft.research.jobs import JobStore

        with JobStore() as store:
            alive = store.worker_is_alive()
        if not alive:
            repo = Path(__file__).resolve().parent.parent
            worker_proc = subprocess.Popen(
                [sys.executable, "-m", "scalper_hft.cli", "job", "worker", "--jobs", "1"],
                cwd=str(repo),
            )
            logger.info("Запущено research worker pid=%s", worker_proc.pid)
    try:
        raise SystemExit(subprocess.call(cmd))
    finally:
        if worker_proc is not None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                worker_proc.kill()


def _enqueue_job(kind: str, params: dict, *, force: bool = False) -> None:
    from scalper_hft.research.jobs import JobStore

    with JobStore() as store:
        job = store.submit(kind, params, force=force)
        print(f"job id={job.id} kind={job.kind} status={job.status} fp={job.short_fp}")
        if job.status == "succeeded" and not force:
            print(f"уже виконано з цими параметрами; повтор — job rerun {job.id}")
        if not store.worker_is_alive():
            print("воркер не запущений: uv run python -m scalper_hft.cli job worker --jobs 2")


def cmd_job(args: argparse.Namespace) -> None:
    from scalper_hft.research.jobs import JobStore

    action = args.job_cmd
    if action == "worker":
        from scalper_hft.research.job_worker import spawn_workers

        spawn_workers(max(1, int(args.jobs)))
        return
    with JobStore() as store:
        if action == "list":
            jobs = store.list_jobs(limit=int(args.limit))
            if not jobs:
                print("черга порожня")
                return
            rows = [
                {
                    "id": j.id,
                    "kind": j.kind,
                    "status": j.status,
                    "fp": j.short_fp,
                    "progress": f"{j.progress_done}/{j.progress_total}" if j.progress_total else "",
                    "error": (j.error or "")[:60],
                }
                for j in jobs
            ]
            print(pd.DataFrame(rows).to_markdown(index=False))
            print("воркер:", "живий" if store.worker_is_alive() else "не запущений")
            return
        if action == "submit":
            import json

            params = json.loads(args.params)
            if not isinstance(params, dict):
                logger.error("--params має бути JSON-об'єктом")
                sys.exit(1)
            job = store.submit(args.kind, params, force=bool(args.force))
            print(f"job id={job.id} kind={job.kind} status={job.status} fp={job.short_fp}")
            return
        if action == "prune":
            stats = store.prune_jobs(
                days=int(args.days),
                status=str(args.status),
                keep_records=bool(args.keep_records),
                dry_run=bool(args.dry_run),
            )
            mode_label = "[DRY-RUN] " if args.dry_run else ""
            print(f"{mode_label}Очищення черги задач (вік >= {args.days} дн., статус: {args.status}):")
            print(f"  Скановано завершених задач:    {stats.scanned_jobs}")
            print(f"  Задач видалено з бази:         {0 if args.keep_records else stats.pruned_jobs}")
            print(f"  Каталогів артефактів видалено: {stats.deleted_dirs} (з них orphaned: {stats.orphaned_dirs})")
            print(f"  Звільнено дискового простору: {stats.freed_mb:.2f} MB ({stats.freed_bytes:,} bytes)")
            return
        job_id = int(args.id)
        entry = store.get(job_id)
        if entry is None:
            logger.error("немає job id=%s", job_id)
            sys.exit(1)
        if action == "status":
            print(
                f"id={entry.id} kind={entry.kind} status={entry.status} fp={entry.fingerprint}\n"
                f"progress={entry.progress_done}/{entry.progress_total} pid={entry.pid}\n"
                f"created={entry.created_at} started={entry.started_at} finished={entry.finished_at}\n"
                f"error={entry.error}"
            )
            tail = store.tail_log(entry.id)
            if tail:
                print("--- log ---")
                print(tail)
            return
        if action == "cancel":
            store.request_cancel(job_id)
            print(f"cancel requested id={job_id}")
            return
        if action == "rerun":
            new_job = store.submit(entry.kind, entry.params, force=True)
            print(f"requeued id={new_job.id} status={new_job.status}")
            return


def cmd_sweep(args: argparse.Namespace) -> None:
    """Матричний прогон: всі стратегії × символи × таймфрейми.

    База (1m) качається один раз на символ; решта таймфреймів — ресемплінг
    з кешу (parquet або PostgreSQL), без повторних звернень до Binance.
    """
    if getattr(args, "enqueue", False):
        from scalper_hft.research.job_handlers import payload_from_sweep_cli

        _enqueue_job("sweep", payload_from_sweep_cli(args))
        return
    from scalper_hft.research.sweep_store import SweepStore
    from scalper_hft.validation.sweep import DEFAULT_INTERVALS, run_sweep, save_sweep_report

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()] if args.strategies else None
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()] if args.intervals else DEFAULT_INTERVALS

    with SweepStore("results/sweep.db") as store:
        df = run_sweep(
            strategies=strategies,
            symbols=symbols,
            intervals=intervals,
            days=args.days,
            base_interval=args.base or "1m",
            mode=args.mode,
            train_bars=args.train,
            test_bars=args.test,
            workers=args.workers,
            include_slow=args.all,
            store=store,
            resume=bool(getattr(args, "resume", True)),
        )

    out_csv = Path(args.out) if args.out else Path("results/sweep.csv")
    out_md = out_csv.with_suffix(".md")
    save_sweep_report(df, out_csv=str(out_csv), out_md=str(out_md))

    ok = df[df["status"] == "ok"].copy()
    print(f"\nSweep: {len(df)} клітинок (ok={len(ok)}, помилок={(df['status'] != 'ok').sum()})")
    print(f"Збережено: {out_csv}, {out_md}\n")
    if ok.empty:
        print("Немає успішних клітинок — перевірте лог помилок.")
        return
    view = ok[["strategy", "symbol", "interval", "n_trades", "total_return", "sharpe", "max_dd", "win_rate"]].copy()
    for c in ["total_return", "max_dd", "win_rate"]:
        view[c] = view[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
    view["sharpe"] = view["sharpe"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "-")
    view = view.sort_values(["interval", "sharpe"], ascending=[True, False])
    print(view.to_markdown(index=False))

    # топ-клітинки за Sharpe у кожному таймфреймі
    print("\n— Топ-3 за Sharpe на таймфрейм —")
    for iv in sorted(df["interval"].unique(), key=lambda x: (len(x), x)):
        sub = ok[ok["interval"] == iv].sort_values("sharpe", ascending=False).head(3)
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            print(
                f"  {iv:>4s}  {r['strategy']:<20s} {r['symbol']:<12s} sharpe={r['sharpe']:+.3f} "
                f"ret={r['total_return']:+.2%} trades={r['n_trades']}"
            )
    if args.notify:
        from scalper_hft.live.telegram import send_telegram

        best = ok.sort_values("sharpe", ascending=False).head(1)
        if not best.empty:
            r = best.iloc[0]
            send_telegram(
                f"📊 Sweep: {len(ok)} ok | топ: {r['strategy']} {r['symbol']} {r['interval']} "
                f"sharpe={r['sharpe']:+.3f} ret={r['total_return']:+.2%}"
            )


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


def _api_bind(
    host: str | None,
    port: int | None,
    *,
    default_host: str,
    default_port: int,
) -> tuple[str, int]:
    """CLI `--host`/`--port` override settings; omit them to keep API_HOST/API_PORT."""
    bind_host = host or default_host
    bind_port = default_port if port is None else port
    if not 1 <= bind_port <= 65535:
        raise ValueError(f"API port must be 1-65535, got {bind_port}")
    return bind_host, bind_port


def cmd_api(args: argparse.Namespace) -> None:
    if args.api_action == "start":
        try:
            import uvicorn
        except ImportError:
            sys.exit("API server requires 'api' extras: uv pip install -e \".[api]\"")
        from scalper_hft.config import get_settings

        settings = get_settings()
        try:
            host, port = _api_bind(
                args.host,
                args.port,
                default_host=settings.api_host,
                default_port=settings.api_port,
            )
        except ValueError as exc:
            sys.exit(str(exc))
        logger.info(f"Запуск FastAPI сервера на {host}:{port}...")
        uvicorn.run("scalper_hft.api.server:app", host=host, port=port, reload=False)

    elif args.api_action == "token":
        try:
            from scalper_hft.api.auth import create_access_token
        except ImportError:
            sys.exit("API server requires 'api' extras: uv pip install -e \".[api]\"")
        token = create_access_token({"sub": "admin"}, expires_delta_hours=args.hours)
        print("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9... [TOKEN GENERATED]")
        print("\nJWT Token (keep it secret!):")
        print(token)
        print("\nДля доступу додайте заголовок: Authorization: Bearer <token>")


def cmd_run(args: argparse.Namespace) -> None:
    from scalper_hft.live.supervisor_config import SupervisorConfig
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor

    cfg = SupervisorConfig.from_yaml(args.config)
    logger.info(f"Запуск LiveTrader з конфігу {args.config} (supervisor: {cfg.name})")

    sup = RegimeSupervisor.from_config(args.config)

    # В реальному коді тут буде виклик LiveTrader або PairsPaperRunner
    # Поки що просто виведемо що ми ініціалізували
    logger.info(f"Ініціалізовано {len(sup._strats)} суб-стратегій")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scalper-hft", description="Високочастотна скальпінг-система (Binance USDT-M)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--exchange", default=None, help="Біржа, e.g. binance, bybit (за замовч. з .env)")
        p.add_argument("--symbol", default=None, help="Символ, e.g. BTCUSDT (за замовч. з .env)")
        p.add_argument("--interval", default=None, help="Таймфрейм: 1s/5s/1m/5m/15m/1h")
        p.add_argument(
            "--strategy",
            default="mean_reversion",
            help="Стратегія з реєстру: mean_reversion, cvd_momentum, pairs_arb, ...",
        )
        p.add_argument("--days", type=int, default=60, help="Глибина історії, днів")
        p.add_argument(
            "--base",
            default="1m",
            help="Базовий таймфрейм для ресемплінгу (за замовч. 1m)",
        )
        p.add_argument(
            "--derive",
            dest="derive",
            action="store_true",
            default=True,
            help="Ресемплити старші таймфрейми з --base (за замовч.)",
        )
        p.add_argument(
            "--no-derive",
            dest="derive",
            action="store_false",
            help="Качати цільовий інтервал з Binance замість ресемплінгу з --base",
        )
        p.add_argument("-p", "--param", action="append", default=[], help="Параметр стратегії: key=value")

    p = sub.add_parser("download", help="Завантажити klines/aggTrades/funding")
    add_common(p)
    p.add_argument("--trades", action="store_true", help="Також aggTrades")
    p.add_argument("--trades-days", type=int, default=None, help="Глибина aggTrades (Binance обмежує 2 доби)")
    p.add_argument("--funding", action="store_true", help="Також funding")
    p.add_argument("--force", action="store_true", help="Оновити хвіст навіть якщо кеш уже покриває період")
    p.add_argument("--retries", type=int, default=None, help="Кількість спроб при помилках/лімітах (за замовч. 8)")
    p.add_argument("--delay", type=float, default=None, help="Затримка між батчами у сек (за замовч. 0.15)")
    p.add_argument(
        "--checkpoint-batches", type=int, default=None, help="Періодичність збереження чекпоінтів (за замовч. 50)"
    )
    p.add_argument("--vision", action="store_true", help="Історичні aggTrades з data.binance.vision")
    p.add_argument("--vision-start", default=None, help="YYYY-MM-DD початок Vision-дампів")
    p.add_argument("--vision-freq", default="daily", choices=["daily", "monthly"])
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("backtest", help="Запустити бектест")
    add_common(p)
    p.add_argument(
        "--breakeven-gate", action="store_true", help="Вимикати сигнали, де очікуваний рух (ATR) < round-trip витрат"
    )
    p.add_argument(
        "--bar-type",
        default="time",
        choices=["time", "dollar", "volume"],
        help="Тип барів для бектесту (time, dollar, volume). Для не-time використовується aggTrades",
    )
    p.add_argument("--bar-threshold", type=float, default=100000.0, help="Поріг для об'ємних або доларових барів")
    p.add_argument("--enqueue", action="store_true", help="Поставити в чергу jobs.sqlite і вийти (не рахувати тут)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("plot", help="Інтерактивний HTML-графік бектесту (свічки+індикатори+угоди+SL/TP)")
    add_common(p)
    p.add_argument("--out", default="docs/plots/backtest.html", help="Шлях до HTML-файлу")
    p.add_argument(
        "--bars", type=int, default=20_000, help="Максимум барів на графіку (даунсемплінг; бари угод зберігаються)"
    )
    p.add_argument("--start", default=None, help="Початок вікна, ISO: YYYY-MM-DD[ HH:MM]")
    p.add_argument("--end", default=None, help="Кінець вікна, ISO: YYYY-MM-DD[ HH:MM]")
    p.add_argument("--no-trades", action="store_true", help="Не малювати точки входу/виходу")
    p.add_argument("--no-sl-tp", action="store_true", help="Не малювати рівні SL/TP")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("walkforward", help="Walk-forward аналіз")
    add_common(p)
    p.add_argument("--train", type=int, default=2000, help="Барів у train (IS)")
    p.add_argument("--test", type=int, default=500, help="Барів у test (OOS)")
    p.add_argument("--use-kalman", action="store_true", help="PairsArb: динамічний Kalman hedge ratio")
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
    p.add_argument("--use-kalman", action="store_true", help="PairsArb: динамічний Kalman hedge ratio")
    p.set_defaults(func=cmd_overfit)

    p = sub.add_parser("cscv", help="PBO через Combinatorial Purged CV")
    add_common(p)
    p.add_argument("--variants", type=int, default=30, help="Кількість випадкових варіантів параметрів")
    p.add_argument("--blocks", type=int, default=8, help="Кількість блоків для розбиття")
    p.add_argument("--max-combos", type=int, default=200, help="Обмеження комбінацій")
    p.set_defaults(func=cmd_cscv)

    p = sub.add_parser(
        "regime-backtest",
        help="Порівняльний бектест: базові стратегії vs RegimeSupervisor",
    )
    p.add_argument(
        "--strategies",
        default="mean_reversion,supertrend,hmm_reversion",
        help="Стратегії через кому (default: mean_reversion,supertrend,hmm_reversion)",
    )
    p.add_argument("--symbol", default="BTCUSDT", help="Символ")
    p.add_argument("--interval", default="1h", help="Таймфрейм")
    p.add_argument("--days", type=int, default=180, help="Кількість днів")
    p.add_argument(
        "--blend-mode",
        default="all",
        choices=["all", "regime_soft", "contextual_hedge", "exp3"],
        help="Режим зважування supervisor-а (default: all — запускає всі три)",
    )
    p.add_argument("--n-hmm-states", type=int, default=3, help="Кількість HMM-станів")
    p.add_argument("--hmm-fit-bars", type=int, default=2000, help="Бари для навчання HMM")
    p.add_argument("--no-regime-table", action="store_true", help="Не виводити таблицю по режимах")
    p.add_argument("--save", action="store_true", help="Зберегти результати у results/")
    p.add_argument("--base", default=None, help="Базовий інтервал для деривації (напр. 1m)")
    p.add_argument("--no-derive", dest="derive", action="store_false", help="Не деривувати ТФ")
    p.set_defaults(func=cmd_regime_backtest, derive=True)

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
    p.add_argument(
        "--use-kalman", action="store_true", help="Динамічний Kalman hedge ratio (дефолт off до OOS bake-off)"
    )
    p.add_argument("--enqueue", action="store_true", help="Поставити в чергу jobs.sqlite і вийти")
    p.set_defaults(func=cmd_pairs)

    p = sub.add_parser("run", help="Запустити trading engine (live/paper) з YAML конфігу")
    p.add_argument("--config", required=True, help="Шлях до YAML файлу (напр. configs/strategies/example.yaml)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pairs-portfolio", help="Бектест портфеля валідованих пар")
    add_common(p)
    p.add_argument("--position-pct", type=float, default=0.3)
    p.add_argument(
        "--method",
        default="equal",
        choices=["equal", "erc"],
        help="Алокація: рівні ваги або Equal Risk Contribution (Narang гл. 6)",
    )
    p.add_argument(
        "--turnover-rate", type=float, default=0.0, help="Штраф за зміну ваг при місячному ребалансі (частка капіталу)"
    )
    p.add_argument("--no-rebalance", action="store_true", help="Без ребалансу ваг")
    p.set_defaults(func=cmd_pairs_portfolio, interval="1h")

    p = sub.add_parser("paper-run-pairs", help="Paper pairs (maker, 2 ноги) або --portfolio")
    add_common(p)
    p.add_argument("--leg1", default="XRPUSDT")
    p.add_argument("--leg2", default="BTCUSDT")
    p.add_argument("--portfolio", action="store_true", help="Три валідовані пари на спільному рахунку")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--sleep", type=int, default=300, help="Пауза між кроками, сек (для 1h — 300+)")
    p.add_argument("--notify", action="store_true")
    p.add_argument(
        "--daemon",
        action="store_true",
        help="Нескінченний цикл до SIGTERM; ігнорує --iterations (для systemd)",
    )
    p.add_argument(
        "--control",
        default="results/control.json",
        help="Шлях до control.json (pause / no_new_entries / flatten)",
    )
    p.set_defaults(func=cmd_paper_run_pairs, strategy="pairs_arb", interval="1h")

    p = sub.add_parser("paper-replay-pairs", help="Історичний paper pairs з моделлю unfilled")
    add_common(p)
    p.add_argument("--leg1", default="XRPUSDT")
    p.add_argument("--leg2", default="BTCUSDT")
    p.add_argument("--notify", action="store_true")
    p.set_defaults(func=cmd_paper_replay_pairs, strategy="pairs_arb", interval="1h")

    p = sub.add_parser("paper-audit", help="Tracking error paper SQLite vs бектест + MAE/MFE forensics")
    p.add_argument("--db", default="results/paper_pairs.sqlite", help="Шлях до paper SQLite")
    p.add_argument("--bt-equity", default=None, help="CSV ts,equity бектесту за той самий період")
    p.add_argument("--bt-fill-rate", type=float, default=None, dest="bt_fill_rate")
    p.add_argument("--dd-mult", type=float, default=1.5, dest="dd_mult", help="Paper maxDD ≤ BT×mult")
    p.set_defaults(func=cmd_paper_audit)

    p = sub.add_parser("experiments", help="Каталог val→OOS експериментів (sparse_basket / ml_strategy)")
    p.add_argument("--catalog", default=None, help="Markdown каталог (за замовч. docs/reports/experiments.md)")
    p.set_defaults(func=cmd_experiments)

    p = sub.add_parser("ml", help="Walk-forward ML-класифікатор: Triple-Barrier + LightGBM + AFML")
    add_common(p)
    p.add_argument(
        "--mode",
        default="triple_barrier",
        choices=["triple_barrier", "horizon"],
        help="Режим лейблінгу: triple_barrier (AFML, default) або horizon",
    )
    p.add_argument("--pt", type=float, default=1.0, help="Profit-take множник (× ATR)")
    p.add_argument("--sl", type=float, default=1.0, help="Stop-loss множник (× ATR)")
    p.add_argument("--holding", type=int, default=10, help="Вертикальний бар'єр (барів)")
    p.add_argument("--decay", type=float, default=0.9, help="Time-decay для sample weights")
    p.add_argument("--frac-d", type=float, default=0.4, dest="frac_d", help="Ступінь fractional differencing")
    p.add_argument("--no-frac-diff", action="store_true", dest="no_frac_diff", help="Вимкнути frac_diff фічі")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--trades", action="store_true", help="Використати aggTrades (CVD + micro фічі)")
    p.add_argument("--hmm", action="store_true", help="Додати HMM-режими (каузальні, без lookahead)")
    p.add_argument("--garch", action="store_true", help="Додати GARCH σ_{t+1} (без lookahead)")
    p.add_argument("--hmm-states", type=int, default=3, dest="hmm_states")
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

    p = sub.add_parser("cohort", help="Cohort analysis: деградація edge за когортами угод")
    add_common(p)
    p.add_argument("--freq", default="ME", help="Частота когорт: ME (місяць), W (тиждень), D")
    p.set_defaults(func=cmd_cohort)

    p = sub.add_parser("lift", help="Децильний lift-аналіз фіч (відбір/фільтри входу)")
    add_common(p)
    p.add_argument("--bins", type=int, default=10, help="Кількість бінів (децилів)")
    p.add_argument("--top", type=int, default=10, help="Скільки топ-фіч показати")
    p.add_argument("--detail", action="store_true", help="Показати повні таблиці lift по фічах")
    p.set_defaults(func=cmd_lift)

    p = sub.add_parser("featimp", help="MDI/MDA/SFI feature importance (AFML Ch.8)")
    add_common(p)
    p.add_argument("--pt", type=float, default=1.0)
    p.add_argument("--sl", type=float, default=1.0)
    p.add_argument("--holding", type=int, default=10)
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--frac-d", type=float, default=0.4, dest="frac_d")
    p.add_argument("--no-frac-diff", action="store_true", dest="no_frac_diff")
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--trades", action="store_true", help="Використати aggTrades (CVD фічі)")
    p.set_defaults(func=cmd_featimp)

    p = sub.add_parser("cfi", help="Clustered Feature Importance (AFML Ch.8.5)")
    add_common(p)
    p.add_argument("--pt", type=float, default=1.0)
    p.add_argument("--sl", type=float, default=1.0)
    p.add_argument("--holding", type=int, default=10)
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--frac-d", type=float, default=0.4, dest="frac_d")
    p.add_argument("--no-frac-diff", action="store_true", dest="no_frac_diff")
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--max-clusters", type=int, default=6, dest="max_clusters")
    p.add_argument("--trades", action="store_true", help="Використати aggTrades (CVD фічі)")
    p.set_defaults(func=cmd_cfi)

    p = sub.add_parser("stress", help="Стрес-тест: crash / liquidity / vol_spike / funding_shock")

    add_common(p)
    p.add_argument("--scenarios", default=None, help="Через кому: crash,liquidity,vol_spike,funding_shock")
    p.set_defaults(func=cmd_stress)

    p = sub.add_parser("capacity", help="Capacity-тест: Sharpe при масштабуванні позицій")
    add_common(p)
    p.add_argument("--scales", default=None, help="Масштаби через кому (напр. 1,2,5,10,20)")
    p.add_argument("--maker", action="store_true", help="Комісії maker")
    p.set_defaults(func=cmd_capacity)

    p = sub.add_parser("survival", help="Survival analysis: час утримання позиції (Kaplan–Meier)")
    add_common(p)
    p.add_argument("--max-time", type=int, default=None, help="Обмежити горизонт (барів)")
    p.add_argument("--top", type=int, default=10, help="Скільки рядків кривої показати")
    p.add_argument("--feature", default=None, help="Фіча для survival_by_feature (напр. atr_14)")
    p.add_argument("--bins", type=int, default=3)
    p.set_defaults(func=cmd_survival)

    p = sub.add_parser("time-decay", help="Time-decay: Sharpe при лагу входу 0..N барів")
    add_common(p)
    p.add_argument("--max-lag", type=int, default=3)
    p.set_defaults(func=cmd_time_decay)

    p = sub.add_parser("quintile", help="Квінтилі z-score спреду (пари)")
    add_common(p)
    p.add_argument("--leg1", default="BTCUSDT")
    p.add_argument("--leg2", default="ETHUSDT")
    p.add_argument("--lookback", type=int, default=240)
    p.set_defaults(func=cmd_quintile, interval="1h")

    p = sub.add_parser("coint-scan", help="Скан коінтеграції символів")
    add_common(p)
    p.add_argument("--symbols", default=None, help="Через кому; інакше DEFAULT_SYMBOLS")
    p.set_defaults(func=cmd_coint_scan, interval="1h")

    p = sub.add_parser("hedge-ratio", help="OOS порівняння 1:1 vs OLS/Johansen hedge")
    add_common(p)
    p.add_argument("--leg1", default="BTCUSDT")
    p.add_argument("--leg2", default="ETHUSDT")
    p.add_argument("--lookback", type=int, default=240)
    p.set_defaults(func=cmd_hedge_ratio, interval="1h")

    p = sub.add_parser(
        "migrate-to-parquet",
        help="Мігрувати всі дані з PostgreSQL у Parquet-файли (data/) і вимкнути postgres-бекенд",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="Символи через кому (за замовч. — всі з Postgres). Приклад: BTCUSDT,ETHUSDT",
    )
    p.add_argument("--overwrite", action="store_true", help="Перезаписати вже наявні Parquet-файли")
    p.add_argument(
        "--skip-trades", action="store_true", dest="skip_trades", help="Не мігрувати aggTrades (великі таблиці)"
    )
    p.add_argument("--skip-funding", action="store_true", dest="skip_funding", help="Не мігрувати funding rates")
    p.add_argument(
        "--data-dir", default=None, dest="data_dir", help="Директорія для Parquet (за замовч. DATA_DIR з .env)"
    )
    p.add_argument("--dry-run", action="store_true", dest="dry_run", help="Показати план міграції без запису файлів")
    p.set_defaults(func=cmd_migrate_to_parquet)

    p = sub.add_parser("mcp", help="Запуск MCP-сервера для трейдінгу (stdio)")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("dashboard", help="Запуск Streamlit-дашборду (інтерпретатор цього venv)")
    p.add_argument("--port", type=int, default=None, help="Порт Streamlit (за замовч. 8501)")
    p.add_argument("--no-worker", action="store_true", help="Не піднімати research worker разом із дашбордом")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("dashboard-hash", help="Згенерувати хеш паролю для DASHBOARD_PASSWORD_HASH")
    p.set_defaults(func=cmd_dashboard_hash)

    job_p = sub.add_parser("job", help="Черга дослідницьких задач (SQLite + worker-процеси)")
    job_sub = job_p.add_subparsers(dest="job_cmd", required=True)
    jw = job_sub.add_parser("worker", help="Довгий процес: claim і виконання job")
    jw.add_argument("--jobs", type=int, default=1, help="Кількість worker-процесів")
    jl = job_sub.add_parser("list", help="Список задач")
    jl.add_argument("--limit", type=int, default=50)
    js = job_sub.add_parser("submit", help="Поставити job з JSON params")
    js.add_argument("kind", choices=["backtest", "pairs", "sweep", "overfit", "capacity"])
    js.add_argument("--params", default="{}", help="JSON-об'єкт параметрів")
    js.add_argument("--force", action="store_true", help="Перезапустити навіть succeeded")
    jst = job_sub.add_parser("status", help="Статус і хвіст логу")
    jst.add_argument("id", type=int)
    jc = job_sub.add_parser("cancel", help="Скасувати queued/running")
    jc.add_argument("id", type=int)
    jr = job_sub.add_parser("rerun", help="Інвалідувати артефакти і поставити в чергу знову")
    jr.add_argument("id", type=int)
    jp = job_sub.add_parser("prune", help="Видалити застарілі завершені jobs та їхні артефакти на диску")
    jp.add_argument("--days", type=int, default=14, help="Вік задач у днях (за замовчуванням: 14)")
    jp.add_argument(
        "--status",
        choices=["all", "succeeded", "failed", "cancelled"],
        default="all",
        help="Фільтр статусів для видалення (за замовчуванням: all)",
    )
    jp.add_argument(
        "--keep-records",
        action="store_true",
        help="Зберегти записи в SQLite, видалити лише важкі артефакти з диска",
    )
    jp.add_argument(
        "--dry-run",
        action="store_true",
        help="Показати, що буде видалено, без фактичного видалення",
    )
    job_p.set_defaults(func=cmd_job)

    p = sub.add_parser(
        "sweep",
        help="Матричний прогон: всі стратегії × символи × таймфрейми (база качається раз, решта — ресемплінг)",
    )
    p.add_argument("--strategies", default=None, help="Через кому; за замовч. — всі single-symbol стратегії")
    p.add_argument("--symbols", default=None, help="Через кому; за замовч. — DEFAULT_SYMBOLS з .env")
    p.add_argument("--intervals", default=None, help="Через кому; за замовч. — 1m,5m,15m,30m,1h,4h")
    p.add_argument("--days", type=int, default=60, help="Глибина історії, днів")
    p.add_argument("--base", default=None, help="Базовий таймфрейм для ресемплінгу (за замовч. 1m)")
    p.add_argument("--mode", default="backtest", choices=["backtest", "walkforward"], help="Режим кожної клітинки")
    p.add_argument("--train", type=int, default=2000, help="Барів train для walkforward")
    p.add_argument("--test", type=int, default=500, help="Барів test (OOS) для walkforward")
    p.add_argument("--workers", type=int, default=1, help="Паралельних клітинок (0/1 = послідовно)")
    p.add_argument("--all", action="store_true", help="Включити ML-стратегію та ensemble (повільно)")
    p.add_argument("--out", default=None, help="Шлях CSV-результату (за замовч. results/sweep.csv)")
    p.add_argument("--notify", action="store_true", help="Telegram-сповіщення топ-результату")
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help="Пропускати вже ok клітинки")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Перерахувати всі клітинки")
    p.add_argument("--enqueue", action="store_true", help="Поставити sweep у чергу jobs.sqlite і вийти")
    p.set_defaults(func=cmd_sweep)

    # ─── telegram-bot ─────────────────────────────────────────────────────────
    tg_p = sub.add_parser(
        "telegram-bot",
        help="Інтерактивний Telegram Bot (команди + push-сповіщення)",
    )
    tg_sub = tg_p.add_subparsers(dest="tg_action", required=True)
    tg_start = tg_sub.add_parser("start", help="Запустити бота (blocking)")
    tg_start.add_argument(
        "--store",
        default=None,
        help="Шлях до SQLite store (за замовч. results/paper_pairs.sqlite)",
    )
    tg_start.add_argument(
        "--control",
        default=None,
        help="Шлях до control.json (за замовч. results/control.json)",
    )
    tg_p.set_defaults(func=cmd_telegram_bot)

    # API
    api_p = sub.add_parser("api", help="FastAPI Server")
    api_sub = api_p.add_subparsers(dest="api_action", required=True)
    api_start = api_sub.add_parser("start", help="Запустити FastAPI сервер (uvicorn)")
    api_start.add_argument(
        "--host",
        default=None,
        help="Хост (за замовч. API_HOST з .env, 0.0.0.0)",
    )
    api_start.add_argument(
        "--port",
        type=int,
        default=None,
        help="Порт (за замовч. API_PORT з .env, 8000)",
    )
    api_token = api_sub.add_parser("token", help="Згенерувати JWT токен для API")
    api_token.add_argument("--hours", type=int, default=24, help="Термін дії токена (годин)")
    api_p.set_defaults(func=cmd_api)

    args = parser.parse_args(argv)
    from scalper_hft.config import get_settings

    settings = get_settings()
    if hasattr(args, "exchange") and args.exchange:
        settings.exchange = args.exchange
    args.param_dict = _parse_param_dict(getattr(args, "param", []))
    args.func(args)


if __name__ == "__main__":
    main()
