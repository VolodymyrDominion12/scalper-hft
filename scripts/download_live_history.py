#!/usr/bin/env python
"""Завантаження РЕАЛЬНОЇ (live) історії для дослідницького кешу.

Ключове: дані качаються з `DATA_EXCHANGE` (live), а не з торгового `EXCHANGE`.
Testnet віддає синтетичну історію — див. `docs/reports/archive_testnet_2026-09-10/README.md`.

Приклад:
    uv run python scripts/download_live_history.py --days 1095
    uv run python scripts/download_live_history.py --symbols BTCUSDT,ETHUSDT --days 1095
    uv run python scripts/download_live_history.py --days 1095 --with-trades   # aggTrades (повільно)

Після завершення обов'язково перевірити:
    uv run python -m scalper_hft.cli data-audit --days 1095
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scalper_hft.config import get_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_live_history")


def _fmt_rows(df) -> str:
    if df is None or df.empty:
        return "0 рядків"
    return f"{len(df)} рядків ({df.index[0]} … {df.index[-1]})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Завантажити live-історію у parquet-кеш")
    parser.add_argument("--symbols", default=None, help="Список через кому (за замовч. DEFAULT_SYMBOLS)")
    parser.add_argument("--days", type=int, default=1095, help="Глибина історії, днів")
    parser.add_argument("--interval", default="1m", help="Базовий таймфрейм klines (за замовч. 1m)")
    parser.add_argument("--with-trades", action="store_true", help="Також aggTrades (REST обмежує ~2 доби)")
    parser.add_argument("--trades-days", type=int, default=2, help="Глибина aggTrades, днів")
    parser.add_argument("--skip-funding", action="store_true", help="Не качати funding")
    parser.add_argument("--batch-delay", type=float, default=None, help="Пауза між батчами, сек")
    parser.add_argument("--checkpoint-batches", type=int, default=None, help="Чекпоінт кожні N батчів")
    args = parser.parse_args()

    settings = get_settings()
    symbols = [s.strip() for s in (args.symbols or ",".join(settings.default_symbols)).split(",") if s.strip()]

    from scalper_hft.config import require_live_data_exchange

    data_exchange = require_live_data_exchange(settings)
    logger.info(
        "Джерело даних: %s [LIVE — НЕ testnet ✓] | торговий EXCHANGE=%s | %d символів × %d днів (%s)",
        data_exchange,
        settings.exchange,
        len(symbols),
        args.days,
        args.interval,
    )

    from scalper_hft.data.downloader import Downloader, download_agg_trades

    dl = Downloader(
        batch_delay=args.batch_delay,
        checkpoint_batches=args.checkpoint_batches,
        exchange_id=data_exchange,
    )

    results: list[tuple[str, str, str, float]] = []
    failed: list[tuple[str, str]] = []
    t_all = time.monotonic()

    for i, sym in enumerate(symbols, 1):
        logger.info("─── [%d/%d] %s ───", i, len(symbols), sym)

        t0 = time.monotonic()
        try:
            kl = dl.klines(sym, args.interval, args.days)
            results.append((sym, f"klines {args.interval}", _fmt_rows(kl), time.monotonic() - t0))
        except Exception as exc:  # noqa: BLE001
            logger.error("klines %s: %s: %s", sym, type(exc).__name__, exc)
            failed.append((sym, f"klines: {type(exc).__name__}: {str(exc)[:150]}"))

        if not args.skip_funding:
            t0 = time.monotonic()
            try:
                fu = dl.funding(sym, args.days)
                results.append((sym, "funding", _fmt_rows(fu), time.monotonic() - t0))
            except Exception as exc:  # noqa: BLE001
                logger.error("funding %s: %s: %s", sym, type(exc).__name__, exc)
                failed.append((sym, f"funding: {type(exc).__name__}: {str(exc)[:150]}"))

        if args.with_trades:
            t0 = time.monotonic()
            try:
                tr = download_agg_trades(sym, args.trades_days, exchange_id=data_exchange)
                results.append((sym, "aggTrades", _fmt_rows(tr), time.monotonic() - t0))
            except Exception as exc:  # noqa: BLE001
                logger.error("aggTrades %s: %s: %s", sym, type(exc).__name__, exc)
                failed.append((sym, f"aggTrades: {type(exc).__name__}: {str(exc)[:150]}"))

    elapsed = time.monotonic() - t_all
    print("\n" + "=" * 78)
    print(f"{'symbol':<10} {'dataset':<14} {'rows':<58} {'sec':>6}")
    print("-" * 78)
    for sym, kind, rows, sec in results:
        print(f"{sym:<10} {kind:<14} {rows[:58]:<58} {sec:6.1f}")
    print("=" * 78)
    print(f"Всього: {len(results)} наборів за {elapsed / 60:.1f} хв")
    if failed:
        print(f"\n❌ Провалено ({len(failed)}):")
        for sym, err in failed:
            print(f"  {sym}: {err}")
        return 1
    print("\n✅ Усі набори завантажено. Далі: uv run python -m scalper_hft.cli data-audit --days ", args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
