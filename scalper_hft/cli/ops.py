"""Операційні команди CLI: download, plot, job, sweep, dashboard, api, mcp, run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from scalper_hft.cli._common import (
    _api_bind,
    _enqueue_job,
    _merge_overlay_params,
    _policy_from_args,
    fail,
    logger,
)
from scalper_hft.config import get_settings


def cmd_download(args: argparse.Namespace) -> None:
    from scalper_hft.data.downloader import download_agg_trades, download_funding, download_klines

    settings = get_settings()
    symbols = args.symbol.split(",") if args.symbol else list(settings.default_symbols)
    intervals = args.interval.split(",") if args.interval else ["1m"]
    retries = getattr(args, "retries", None)
    batch_delay = getattr(args, "delay", None)
    checkpoint_batches = getattr(args, "checkpoint_batches", None)
    # Ринкові дані: --exchange override, інакше DATA_EXCHANGE (не торговий EXCHANGE —
    # той може бути testnet і віддати синтетичну історію).
    exchange_id = getattr(args, "exchange", None) or settings.data_exchange
    is_testnet = "testnet" in str(exchange_id).lower()
    env_label = "УВАГА: TESTNET ⚠ (синтетичні дані)" if is_testnet else "LIVE (НЕ testnet ✓)"
    logger.info(
        "Джерело історичних даних: %s [%s] (налаштування: DATA_EXCHANGE=%s, торговий EXCHANGE=%s)",
        exchange_id,
        env_label,
        settings.data_exchange,
        settings.exchange,
    )

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
            if df is not None and not df.empty:
                logger.info("klines %s %s: %d свічок (%s … %s)", sym, iv, len(df), df.index[0], df.index[-1])
            else:
                logger.info("klines %s %s: 0 свічок", sym, iv)
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

    params = dict(args.param_dict)
    overlay = _policy_from_args(args)
    if overlay is not None:
        params = _merge_overlay_params(args, params, overlay)
    strategy = get_strategy(args.strategy, **params)
    settings = get_settings()
    bundle = load_research_data(
        args.symbol,
        args.interval,
        args.days,
        strategy,
        base=getattr(args, "base", None) or "1m",
        derive=getattr(args, "derive", True),
        exchange_id=getattr(args, "exchange", None) or settings.data_exchange,
    )
    df = bundle.klines
    if df is None or df.empty:
        fail("Немає даних %s %s — запустіть download спершу", args.symbol, args.interval)
    cost = CostModel(maker_fee=settings.maker_fee, taker_fee=settings.taker_fee, slippage_frac=settings.slippage_frac)
    res = run_strategy_backtest(
        df,
        strategy,
        cost=cost,
        trades=bundle.trades,
        funding=bundle.funding,
        position_pct=settings.position_pct,
        overlay=overlay,
        interval=args.interval or settings.default_interval,
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


def cmd_macro_recorder(args: argparse.Namespace) -> None:
    """Запис ліквідацій та OI у реальному часі."""
    from scalper_hft.live.macro_recorder import record_macro

    for sym in (args.symbol or "BTCUSDT").split(","):
        record_macro(sym, minutes=args.minutes)
        print(f"{sym}: запущено macro_recorder на {args.minutes} хвилин")


def cmd_download_liquidations(args: argparse.Namespace) -> None:
    from scalper_hft.data.downloader import download_liquidations

    logger.info(
        "Джерело ліквідацій: Binance Vision Archives [LIVE (НЕ testnet ✓)] (https://data.binance.vision/data/futures/um)"
    )
    for sym in (args.symbol or "BTCUSDT").split(","):
        download_liquidations(sym, args.days)


def cmd_download_oi(args: argparse.Namespace) -> None:
    from scalper_hft.config import get_settings
    from scalper_hft.data.downloader import download_oi

    settings = get_settings()
    exchange_id = getattr(args, "exchange", None) or settings.data_exchange
    is_testnet = "testnet" in str(exchange_id).lower()
    env_label = "УВАГА: TESTNET ⚠ (синтетичні дані)" if is_testnet else "LIVE (НЕ testnet ✓)"
    logger.info("Джерело Open Interest: %s [%s]", exchange_id, env_label)

    for sym in (args.symbol or "BTCUSDT").split(","):
        download_oi(sym, args.days, exchange_id=exchange_id)


def cmd_data_audit(args: argparse.Namespace) -> None:
    """Аудит кешу ринкових даних проти LIVE-біржі (`DATA_EXCHANGE`).

    Fail-closed: exit code 1, якщо хоч один символ не пройшов — щоб команду
    можна було ставити у cron/CI перед дослідницькими прогонами.
    """
    import json

    from scalper_hft.data.audit import audit_symbols, format_report

    settings = get_settings()
    symbols = [s.strip() for s in (args.symbol or ",".join(settings.default_symbols)).split(",") if s.strip()]
    interval = args.interval or "1m"
    results = audit_symbols(
        symbols,
        interval=interval,
        days=args.days,
        live_samples=int(getattr(args, "live_samples", 5)),
        check_funding=not getattr(args, "no_funding", False),
    )
    report = format_report(results, days=args.days)
    print(report)
    if getattr(args, "json", None):
        Path(args.json).write_text(
            json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("JSON-звіт: %s", args.json)
    bad = [r for r in results if not r.ok]
    if bad:
        fail(f"Аудит даних провалено для {len(bad)} символ(ів): {', '.join(r.symbol for r in bad)}")


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

    from scalper_hft.config import get_settings, require_dashboard_password

    settings = get_settings()
    bind = args.address or settings.dashboard_host or "127.0.0.1"
    try:
        require_dashboard_password(bind, settings)
    except RuntimeError as exc:
        fail(str(exc))

    script = Path(__file__).resolve().parent / "dashboard.py"
    try:
        import streamlit  # noqa: F401
    except ImportError:
        fail("Немає streamlit. Встановіть: uv pip install -e '.[dashboard]'")
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), "--server.address", bind]
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
                fail("--params має бути JSON-об'єктом")
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
        if action in {"delete", "rm"}:
            job_ids = [int(i) for i in args.ids]
            count = store.delete_jobs(job_ids)
            print(f"Видалено задач з бази та очищено папок артефактів з диска: {count}")
            return
        job_id = int(args.id)
        entry = store.get(job_id)
        if entry is None:
            logger.error("немає job id=%s", job_id)
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
        overlay_book = None
        if getattr(args, "overlay", None):
            from scalper_hft.overlay import load_overlay_book

            overlay_book = load_overlay_book(args.overlay)
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
            overlay_book=overlay_book,
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


def cmd_api(args: argparse.Namespace) -> None:
    if args.api_action == "start":
        try:
            import uvicorn
        except ImportError:
            fail("API server requires 'api' extras: uv pip install -e \".[api]\"")
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
            fail(str(exc))
        from scalper_hft.config import require_safe_api_bind

        try:
            require_safe_api_bind(host, settings)
        except RuntimeError as exc:
            fail(str(exc))
        logger.info(f"Запуск FastAPI сервера на {host}:{port}...")
        uvicorn.run("scalper_hft.api.server:app", host=host, port=port, reload=False)

    elif args.api_action == "token":
        try:
            from scalper_hft.api.auth import create_access_token
        except ImportError:
            fail("API server requires 'api' extras: uv pip install -e \".[api]\"")
        token = create_access_token({"sub": "admin"}, expires_delta_hours=args.hours)
        print("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9... [TOKEN GENERATED]")
        print("\nJWT Token (keep it secret!):")
        print(token)
        print("\nДля доступу додайте заголовок: Authorization: Bearer <token>")


def cmd_run(args: argparse.Namespace) -> None:
    """Dry-init перевірка YAML конфіга supervisor (НЕ запускає торгівлю).

    Реальний запуск: paper-run / paper-run-pairs (paper) або live-деплой
    через systemd на VPS (docs/DEPLOY_PLAN.md). Ця команда лише валідує,
    що конфіг парситься і суб-стратегії ініціалізуються.
    """
    from scalper_hft.live.supervisor_config import SupervisorConfig
    from scalper_hft.strategies.regime_supervisor import RegimeSupervisor

    cfg = SupervisorConfig.from_yaml(args.config)
    sup = RegimeSupervisor.from_config(args.config)
    print(f"✅ Конфіг {args.config} валідний (supervisor: {cfg.name})")
    print(f"   суб-стратегій ініціалізовано: {len(sup._strats)}")
    print("   Це dry-init перевірка — торгівля НЕ запускається.")
    print("   Paper: paper-run-pairs --portfolio | Live: docs/DEPLOY_PLAN.md (systemd)")


def cmd_is_report(args: argparse.Namespace) -> None:
    """Генерація Implementation Shortfall (IS) звіту."""
    from scalper_hft.config import get_settings
    from scalper_hft.live.is_report import build_from_orders, calibrate_slippage_bps
    from scalper_hft.live.store import PaperStore

    settings = get_settings()
    days = int(getattr(args, "days", 7) or 7)

    with PaperStore() as store:
        orders = store.recent_orders(limit=10_000)
    if orders is None or orders.empty:
        print("Немає ордерів для аналізу IS.")
        return
    if "ts" in orders.columns:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        orders = orders[pd.to_datetime(orders["ts"], utc=True) >= cutoff]

    report = build_from_orders(orders, model_slippage_bps=settings.slippage_bps)
    print(report.summary())
    if report.coverage_ok:
        rec = calibrate_slippage_bps(report)
        print(f"  Рекомендований SLIPPAGE_BPS (з IS): {rec:.2f}")
    else:
        print(f"  Потрібно ≥20 fills для калібровки (зараз {report.n_fills})")
