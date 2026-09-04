"""Виконавці job: backtest, pairs, sweep. Без subprocess CLI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scalper_hft.research.job_artifacts import save_backtest_result
from scalper_hft.research.jobs import JobStore

logger = logging.getLogger(__name__)

Handler = Callable[..., None]


def handle_backtest(payload: dict[str, Any], job_dir: Path, **_: Any) -> None:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.downloader import download_agg_trades, download_funding
    from scalper_hft.strategies import get_strategy

    settings = get_settings()
    name = str(payload["strategy"])
    symbol = str(payload["symbol"])
    interval = str(payload["interval"])
    days = int(payload["days"])
    params = dict(payload.get("params") or {})
    if payload.get("breakeven_gate"):
        params["breakeven_gate"] = True
    trace = bool(payload.get("trace", False))
    is_maker = bool(payload.get("maker", False))
    base = str(payload.get("base_interval") or "1m")
    strategy = get_strategy(name, **params)
    df = ensure_klines(symbol, interval, days, base_interval=base, derive=True)
    if df is None or df.empty:
        raise RuntimeError(f"немає даних {symbol} {interval}")
    trades = download_agg_trades(symbol, days) if getattr(strategy, "needs_trades", False) else None
    funding = download_funding(symbol, days) if getattr(strategy, "needs_funding", False) else None
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    res = run_strategy_backtest(
        df,
        strategy,
        cost=cost,
        trades=trades,
        funding=funding,
        position_pct=settings.position_pct,
        is_maker=is_maker,
        trace=trace,
    )
    save_backtest_result(job_dir, res, extra={"symbol": symbol, "interval": interval, "days": days})
    logger.info("backtest %s %s %s: %s", name, symbol, interval, res.metrics.summary().splitlines()[0])


def handle_pairs(payload: dict[str, Any], job_dir: Path, **_: Any) -> None:
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.pairs import run_pairs_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.downloader import download_funding
    from scalper_hft.strategies import get_strategy

    settings = get_settings()
    name = str(payload.get("strategy") or "pairs_arb")
    leg1 = str(payload["leg1"])
    leg2 = str(payload["leg2"])
    interval = str(payload["interval"])
    days = int(payload["days"])
    params = dict(payload.get("params") or {})
    maker = bool(payload.get("maker", True))
    base = str(payload.get("base_interval") or "1m")
    position_pct = float(payload.get("position_pct") or settings.pair_notional_pct)
    strategy = get_strategy(name, **params)
    d1 = ensure_klines(leg1, interval, days, base_interval=base, derive=True)
    d2 = ensure_klines(leg2, interval, days, base_interval=base, derive=True)
    if d1 is None or d2 is None or d1.empty or d2.empty:
        raise RuntimeError(f"немає даних {leg1}/{leg2} {interval}")
    f1 = download_funding(leg1, days)
    f2 = download_funding(leg2, days)
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    res = run_pairs_backtest(
        d1,
        d2,
        strategy,
        f1,
        f2,
        position_pct=position_pct,
        cost=cost,
        maker_execution=maker,
    )
    save_backtest_result(
        job_dir,
        res,
        extra={"leg1": leg1, "leg2": leg2, "interval": interval, "days": days, "maker": maker},
    )
    logger.info("pairs %s/%s %s: ret=%s", leg1, leg2, interval, res.metrics.total_return)


def handle_sweep(
    payload: dict[str, Any], job_dir: Path, *, store_path: str | Path | None = None, job_id: int | None = None
) -> None:
    from scalper_hft.research.sweep_store import SweepStore
    from scalper_hft.validation.sweep import run_sweep, save_sweep_report

    strategies = list(payload.get("strategies") or [])
    symbols = list(payload.get("symbols") or [])
    intervals = list(payload.get("intervals") or [])
    days = int(payload.get("days") or 60)
    mode = str(payload.get("mode") or "backtest")
    workers = int(payload.get("workers") or 1)
    resume = bool(payload.get("resume", True))
    enable_trace = bool(payload.get("enable_trace", False))
    include_slow = bool(payload.get("include_slow", False))
    base = str(payload.get("base_interval") or "1m")
    train_bars = int(payload.get("train_bars") or 2000)
    test_bars = int(payload.get("test_bars") or 500)

    sweep_db = Path(payload.get("sweep_db") or "results/sweep.db")
    jobs_path = Path(store_path) if store_path else None

    def on_progress(done: int, total: int) -> None:
        if jobs_path is None or job_id is None:
            return
        with JobStore(jobs_path) as js:
            js.set_progress(job_id, done, total)

    store = SweepStore(sweep_db)
    try:
        df = run_sweep(
            strategies=strategies or None,
            symbols=symbols or None,
            intervals=intervals or None,
            days=days,
            base_interval=base,
            mode=mode,
            train_bars=train_bars,
            test_bars=test_bars,
            workers=workers,
            include_slow=include_slow,
            store=store,
            resume=resume,
            enable_trace=enable_trace,
            on_progress=on_progress,
        )
    finally:
        store.close()
    job_dir.mkdir(parents=True, exist_ok=True)
    out_csv = job_dir / "sweep.csv"
    save_sweep_report(df, out_csv=str(out_csv), out_md=str(out_csv.with_suffix(".md")))
    logger.info("sweep: %d рядків → %s", len(df), out_csv)


def handle_test_sleep(payload: dict[str, Any], job_dir: Path, **_: Any) -> None:
    """Лише для тестів cancel/heartbeat (не в CLI)."""
    import time

    job_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(float(payload.get("seconds", 2)))


HANDLERS: dict[str, Handler] = {
    "backtest": handle_backtest,
    "pairs": handle_pairs,
    "sweep": handle_sweep,
    "_test_sleep": handle_test_sleep,
}


def run_job(
    kind: str,
    payload: dict[str, Any],
    job_dir: Path,
    *,
    store_path: str | Path | None = None,
    job_id: int | None = None,
) -> None:
    fn = HANDLERS.get(kind)
    if fn is None:
        raise KeyError(f"невідомий kind '{kind}' (доступні: {sorted(HANDLERS)})")
    fn(payload, job_dir, store_path=store_path, job_id=job_id)


def payload_from_backtest_cli(args: Any) -> dict[str, Any]:
    from scalper_hft.config import get_settings

    settings = get_settings()
    symbol = args.symbol or (settings.default_symbols[0] if settings.default_symbols else "BTCUSDT")
    interval = args.interval or settings.default_interval
    return {
        "strategy": args.strategy,
        "symbol": symbol,
        "interval": interval,
        "days": args.days,
        "params": dict(getattr(args, "param_dict", None) or {}),
        "maker": bool(getattr(args, "maker", False)),
        "trace": bool(getattr(args, "trace", False)),
        "base_interval": getattr(args, "base", None) or "1m",
        "breakeven_gate": bool(getattr(args, "breakeven_gate", False)),
    }


def payload_from_pairs_cli(args: Any) -> dict[str, Any]:
    from scalper_hft.config import get_settings

    settings = get_settings()
    params = dict(getattr(args, "param_dict", None) or {})
    if getattr(args, "use_kalman", False):
        params["use_kalman"] = True
    pct = getattr(args, "position_pct", None)
    if pct is None:
        pct = settings.pair_notional_pct
    return {
        "strategy": args.strategy,
        "leg1": args.leg1,
        "leg2": args.leg2,
        "interval": args.interval or "1h",
        "days": args.days,
        "params": params,
        "maker": bool(getattr(args, "maker", False)),
        "position_pct": float(pct),
        "base_interval": getattr(args, "base", None) or "1m",
    }


def payload_from_sweep_cli(args: Any) -> dict[str, Any]:
    def _split(raw: str | None) -> list[str]:
        if not raw:
            return []
        return [s.strip() for s in raw.split(",") if s.strip()]

    return {
        "strategies": _split(getattr(args, "strategies", None)),
        "symbols": _split(getattr(args, "symbols", None)),
        "intervals": _split(getattr(args, "intervals", None)),
        "days": int(args.days),
        "mode": args.mode,
        "workers": int(args.workers),
        "resume": bool(getattr(args, "resume", True)),
        "enable_trace": False,
        "include_slow": bool(getattr(args, "all", False)),
        "base_interval": getattr(args, "base", None) or "1m",
        "train_bars": int(args.train),
        "test_bars": int(args.test),
    }
