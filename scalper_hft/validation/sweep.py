"""Матричний прогон гіпотез: всі стратегії × символи × таймфрейми.

Мета (дослідження: перевірка багатьох гіпотез без частих звернень до API):
    - база (1m) качається з Binance один раз на символ і кешується
      (parquet або PostgreSQL у Docker);
    - решта таймфреймів (5m/15m/30m/1h/4h/1d...) будується ресемплінгом
      локально — нуль API-дзвінків;
    - кожна клітинка (стратегія, символ, таймфрейм) — незалежний бектест
      або walk-forward з повними комісіями; результати зберігаються у SQLite.

Результат — таблиця `SweepRow` (з sweep_store): strategy, symbol, interval,
n_bars, n_trades, total_return, sharpe, sortino, calmar, max_dd, win_rate,
profit_factor, avg_trade, exposure, trades_per_day
(+ OOS-метрики у режимі walkforward). Помилки однієї клітинки не зупиняють
прогон — вони фіксуються у status/error.

Resuming:
    store = SweepStore("results/sweep.db")
    df = run_sweep(..., store=store, resume=True)  # пропускає вже виконані
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from scalper_hft.backtest.execution import CostModel
from scalper_hft.research.sweep_store import SweepRow, SweepStore

logger = logging.getLogger(__name__)


# Двоногі стратегії (потребують пару/кошик символів) — у пер-символьному
# sweep не мають сенсу; ML/ensemble — повільні, включаються лише за запитом.
MULTI_SYMBOL_STRATEGIES = frozenset({"pairs_arb", "sparse_basket", "funding_arb"})
SLOW_STRATEGIES = frozenset({"ml_strategy", "ensemble"})

DEFAULT_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h"]


def default_strategies(*, include_slow: bool = False) -> list[str]:
    """Стратегії для sweep за замовчуванням: всі single-symbol без ML/ensemble."""
    from scalper_hft.strategies import REGISTRY

    names = sorted(set(REGISTRY) - MULTI_SYMBOL_STRATEGIES)
    if not include_slow:
        names = [n for n in names if n not in SLOW_STRATEGIES]
    return names


def _run_backtest_cell(
    strategy: Any,
    symbol: str,
    interval: str,
    klines: pd.DataFrame,
    trades: pd.DataFrame | None,
    funding: pd.DataFrame | None,
    *,
    cost: CostModel,
    position_pct: float,
    enable_trace: bool = False,
) -> SweepRow:
    from scalper_hft.backtest.router import run_strategy_backtest

    res = run_strategy_backtest(
        klines,
        strategy,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=position_pct,
        trace=enable_trace,
    )
    m = res.metrics

    # Filter attribution з трейсу
    n_raw = 0
    n_filtered = 0
    filter_attr_json = ""
    if enable_trace and res.trace is not None:
        from scalper_hft.research.filter_trace import filter_attribution

        n_raw = len(res.trace)
        n_filtered = res.trace.n_blocked()
        attr_df = filter_attribution(res.trace)
        if not attr_df.empty:
            filter_attr_json = json.dumps(dict(zip(attr_df["filter_name"], attr_df["n_blocked"].tolist())))

    row = SweepRow(
        strategy=strategy.name,
        symbol=symbol,
        interval=interval,
        n_bars=len(klines),
        n_trades=int(m.n_trades),
        total_return=float(m.total_return),
        sharpe=float(m.sharpe),
        sortino=float(m.sortino),
        calmar=float(m.calmar),
        max_dd=float(m.max_drawdown),
        win_rate=float(m.win_rate),
        profit_factor=float(m.profit_factor),
        avg_trade=float(m.avg_trade_return),
        exposure=float(m.exposure),
        trades_per_day=float(m.trades_per_day),
        n_raw_signals=n_raw,
        n_filtered=n_filtered,
        filter_attribution=filter_attr_json,
    )
    return row


def _run_wf_cell(
    strategy: Any,
    symbol: str,
    interval: str,
    klines: pd.DataFrame,
    trades: pd.DataFrame | None,
    funding: pd.DataFrame | None,
    *,
    train_bars: int,
    test_bars: int,
    cost: CostModel,
    position_pct: float,
) -> SweepRow:
    from scalper_hft.validation.walk_forward import run_walk_forward

    res = run_walk_forward(
        klines,
        strategy,
        train_bars,
        test_bars,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=position_pct,
    )
    return SweepRow(
        strategy=strategy.name,
        symbol=symbol,
        interval=interval,
        n_bars=len(klines),
        n_trades=sum(w.n_trades for w in res.windows),
        total_return=float("nan"),
        sharpe=float(res.avg_oos_sharpe),
        max_dd=float("nan"),
        win_rate=float("nan"),
        profit_factor=float("nan"),
        avg_trade=float("nan"),
        exposure=float("nan"),
        avg_is_sharpe=float(res.avg_is_sharpe),
        avg_oos_sharpe=float(res.avg_oos_sharpe),
        oos_positive_frac=float(res.positive_windows_frac),
    )


def _build_cell_runner(
    *,
    days: int,
    base_interval: str,
    mode: str,
    train_bars: int,
    test_bars: int,
    cost: CostModel,
    position_pct: float,
    data_provider: Callable[..., Any] | None,
    enable_trace: bool = False,
):
    """Повертає функцію клітинки (strategy_name, symbol, interval) -> SweepRow."""
    from scalper_hft.strategies import get_strategy

    def cell(name: str, symbol: str, interval: str) -> SweepRow:
        strategy = get_strategy(name)
        if data_provider is not None:
            klines, trades, funding = data_provider(symbol, interval, days)
        else:
            from scalper_hft.data.access import ensure_klines
            from scalper_hft.data.downloader import download_agg_trades, download_funding

            klines = ensure_klines(symbol, interval, days, base_interval=base_interval, derive=True)
            trades = download_agg_trades(symbol, days) if strategy.needs_trades else None
            funding = download_funding(symbol, days) if strategy.needs_funding else None
        if klines is None or klines.empty:
            return SweepRow(
                strategy=strategy.name, symbol=symbol, interval=interval, status="error", error="немає даних"
            )
        if mode == "walkforward":
            return _run_wf_cell(
                strategy,
                symbol,
                interval,
                klines,
                trades,
                funding,
                train_bars=train_bars,
                test_bars=test_bars,
                cost=cost,
                position_pct=position_pct,
            )
        return _run_backtest_cell(
            strategy,
            symbol,
            interval,
            klines,
            trades,
            funding,
            cost=cost,
            position_pct=position_pct,
            enable_trace=enable_trace,
        )

    return cell


def run_sweep(
    strategies: list[str] | None = None,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    days: int = 60,
    *,
    base_interval: str = "1m",
    mode: str = "backtest",
    train_bars: int = 2000,
    test_bars: int = 500,
    workers: int = 1,
    include_slow: bool = False,
    data_provider: Callable[..., Any] | None = None,
    store: SweepStore | None = None,
    resume: bool = False,
    enable_trace: bool = False,
) -> pd.DataFrame:
    """Прогнати матрицю стратегій × символів × таймфреймів.

    store: SweepStore для persistence (upsert кожної клітинки).
    resume: якщо True і store задано — пропускати вже виконані комбінації.
    enable_trace: якщо True — збирати filter attribution для кожної клітинки.

    Returns:
        DataFrame з рядками SweepRow (одна клітинка = один рядок).
    """
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import REGISTRY

    settings = get_settings()
    strategies = strategies or default_strategies(include_slow=include_slow)
    symbols = symbols or list(settings.default_symbols)
    intervals = intervals or DEFAULT_INTERVALS
    for name in strategies:
        if name not in REGISTRY:
            raise KeyError(f"Невідома стратегія '{name}' (доступні: {sorted(REGISTRY)})")

    # прогріти базовий таймфрейм послідовно — один API-прохід на символ,
    # щоб паралельні клітинки не качали базу одночасно (гонка на запис
    # Postgres = duplicate key / повторні завантаження тих самих вікон).
    # Прогрів потрібен і коли 1m немає у списку інтервалів (5m/15m/1h
    # ресемпляться з бази), тому перевіряємо can_derive, а не `in intervals`.
    if data_provider is None:
        from scalper_hft.data.access import can_derive, warm_base_cache

        need_base = any(iv == base_interval or can_derive(iv, base_interval) for iv in intervals)
        if need_base:
            logger.info("Прогрів бази %s для %s символів (%d днів)...", base_interval, len(symbols), days)
            warm_base_cache(symbols, days, base_interval=base_interval)
        need_funding = any(getattr(REGISTRY[n], "needs_funding", False) for n in strategies)
        if need_funding:
            from scalper_hft.data.downloader import download_funding

            logger.info("Прогрів funding для %s символів...", len(symbols))
            for sym in symbols:
                download_funding(sym, days)

    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    cell = _build_cell_runner(
        days=days,
        base_interval=base_interval,
        mode=mode,
        train_bars=train_bars,
        test_bars=test_bars,
        cost=cost,
        position_pct=settings.position_pct,
        data_provider=data_provider,
        enable_trace=enable_trace,
    )

    all_cells = [(name, sym, iv) for sym in symbols for iv in intervals for name in strategies]

    # -- Resume: фільтрувати вже виконані клітинки ----------------------------
    if resume and store is not None:
        cells = [c for c in all_cells if not store.already_done(c[0], c[1], c[2], days, mode)]
        skipped = len(all_cells) - len(cells)
        if skipped:
            logger.info("Resume: пропущено %d вже виконаних клітинок", skipped)
    else:
        cells = all_cells

    total = len(cells)
    logger.info(
        "Sweep: %d стратегій × %d символів × %d TF = %d клітинок (всього %d)",
        len(strategies),
        len(symbols),
        len(intervals),
        total,
        len(all_cells),
    )

    # -- tqdm progress bar (якщо встановлений) --------------------------------
    try:
        from tqdm import tqdm

        progress_iter = tqdm(total=total, desc="Sweep", unit="cell")
    except ImportError:
        progress_iter = None

    def _run_and_store(c: tuple) -> SweepRow:
        """Виконати клітинку, зберегти у store, оновити прогрес."""
        try:
            row = cell(*c)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Клітинка %s %s %s: %s", *c, exc)
            row = SweepRow(
                strategy=c[0], symbol=c[1], interval=c[2], days=days, mode=mode, status="error", error=str(exc)
            )
        else:
            row.days = days
            row.mode = mode
        if store is not None:
            store.upsert(row)
        if progress_iter is not None:
            progress_iter.update(1)
        return row

    rows: list[SweepRow] = []
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_and_store, c): c for c in cells}
            for fut in as_completed(futs):
                try:
                    rows.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    c = futs[fut]
                    logger.warning("Future %s %s %s: %s", *c, exc)
    else:
        for c in cells:
            rows.append(_run_and_store(c))

    if progress_iter is not None:
        progress_iter.close()

    result_df = pd.DataFrame([r.as_dict() for r in rows])

    # -- Якщо є store — повертаємо всі результати (включно зі скіпнутими) ----
    if store is not None and resume:
        try:
            result_df = store.load()
        except Exception:  # noqa: BLE001
            pass

    return result_df


def save_sweep_report(df: pd.DataFrame, out_csv: str | None = None, out_md: str | None = None) -> None:
    """Зберегти результати sweep у CSV і короткий markdown-звіт."""
    from pathlib import Path

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.info("Sweep CSV: %s", out_csv)

    if out_md:
        ok = df[df["status"] == "ok"]
        md = ["# Sweep: стратегії × символи × таймфрейми", ""]
        md.append(f"- клітинок: {len(df)} (ok: {len(ok)}, помилок: {(df['status'] != 'ok').sum()})")
        md.append("")
        for iv in sorted(df["interval"].unique(), key=lambda x: (len(x), x)):
            sub = ok[ok["interval"] == iv]
            if sub.empty:
                continue
            md.append(f"## {iv}")
            md.append("")
            cols = [
                "strategy",
                "symbol",
                "n_trades",
                "total_return",
                "sharpe",
                "max_dd",
                "win_rate",
                "avg_oos_sharpe",
                "oos_positive_frac",
            ]
            view = sub[[c for c in cols if c in sub.columns]].copy()
            for c in ["total_return", "max_dd", "win_rate"]:
                if c in view.columns:
                    view[c] = view[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
            view["sharpe"] = view["sharpe"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "-")
            if "avg_oos_sharpe" in view.columns:
                view["avg_oos_sharpe"] = view["avg_oos_sharpe"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "-")
            if "oos_positive_frac" in view.columns:
                view["oos_positive_frac"] = view["oos_positive_frac"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
            md.append(view.to_markdown(index=False))
            md.append("")
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text("\n".join(md), encoding="utf-8")
        logger.info("Sweep звіт: %s", out_md)


__all__ = ["SweepRow", "SweepStore", "run_sweep", "save_sweep_report", "default_strategies", "DEFAULT_INTERVALS"]
