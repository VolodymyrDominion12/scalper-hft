"""Міграція даних PostgreSQL → Parquet.

Читає всі klines / aggTrades / funding з PostgresStore і зберігає їх
у ParquetStore (data/), зберігаючи ту саму структуру файлів, що вже
використовує вся система.

Використання (standalone):
    uv run python scripts/migrate_postgres_to_parquet.py

Використання (через CLI):
    uv run python -m scalper_hft.cli migrate-to-parquet

Опції:
    --symbol SYM1,SYM2   Мігрувати лише вказані символи (за замовч. — всі з Postgres)
    --overwrite          Перезаписати вже наявні Parquet-файли
    --skip-trades        Не мігрувати aggTrades (великі таблиці)
    --skip-funding       Не мігрувати funding
    --data-dir PATH      Куди писати parquet (за замовч. DATA_DIR з .env / ./data)
    --dry-run            Показати, що буде мігровано, без запису файлів

Ідемпотентність: без --overwrite вже наявні parquet-файли пропускаються,
тому скрипт можна переривати і запускати повторно.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── bootstrap path так, щоб запуск standalone теж працював ──────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scalper_hft.config import get_settings
from scalper_hft.data.storage import (
    funding_path,
    klines_path,
    save_funding,
    save_klines,
    save_trades,
    trades_path,
)
from scalper_hft.data.store import PostgresStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("migrate")


def _fmt_rows(n: int) -> str:
    return f"{n:,}".replace(",", "_")


def migrate(
    *,
    symbols: list[str] | None = None,
    overwrite: bool = False,
    skip_trades: bool = False,
    skip_funding: bool = False,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Основна функція міграції. Повертає {ключ: кількість рядків} для звіту.

    Args:
        symbols: Список символів для міграції. None = всі з Postgres.
        overwrite: Перезаписати вже наявні Parquet-файли.
        skip_trades: Пропустити міграцію aggTrades.
        skip_funding: Пропустити міграцію funding.
        data_dir: Директорія для збереження Parquet (за замовч. з Settings).
        dry_run: Тільки показати план, без запису.
    """
    settings = get_settings()
    out_dir = data_dir or settings.data_dir_abs
    out_dir.mkdir(parents=True, exist_ok=True)

    pg = PostgresStore()

    # ── 1. Отримати список (symbol, interval) з Postgres ─────────────────────
    logger.info("Зчитую список klines з Postgres...")
    try:
        all_klines = pg.list_klines()
    except Exception as exc:  # noqa: BLE001
        logger.error("Не вдалося підключитись до Postgres: %s", exc)
        logger.error("Переконайтесь, що контейнер запущений: docker compose up -d postgres")
        sys.exit(1)

    if not all_klines:
        logger.warning("У Postgres немає жодної пари (symbol, interval) у таблиці klines.")

    # ── 2. Фільтрація за --symbol ─────────────────────────────────────────────
    if symbols:
        sym_set = {s.upper() for s in symbols}
        all_klines = [(s, iv) for s, iv in all_klines if s.upper() in sym_set]
        logger.info("Фільтровано до %d пар: %s", len(all_klines), sym_set)

    # Збираємо унікальні символи для trades/funding
    all_symbols = sorted({s for s, _ in all_klines})

    # ── 3. Dry-run: показати план і вийти ────────────────────────────────────
    if dry_run:
        logger.info("=== DRY RUN — нічого не записується ===")
        logger.info("klines пар для міграції: %d", len(all_klines))
        for sym, iv in all_klines:
            path = klines_path(out_dir, sym, iv)
            status = "є" if path.exists() else "відсутній"
            logger.info("  klines %s %s -> %s  [%s]", sym, iv, path.name, status)
        if not skip_trades:
            for sym in all_symbols:
                path = trades_path(out_dir, sym)
                status = "є" if path.exists() else "відсутній"
                logger.info("  aggTrades %s -> %s  [%s]", sym, path.name, status)
        if not skip_funding:
            for sym in all_symbols:
                path = funding_path(out_dir, sym)
                status = "є" if path.exists() else "відсутній"
                logger.info("  funding %s -> %s  [%s]", sym, path.name, status)
        return {}

    stats: dict[str, int] = {}
    t0_total = time.monotonic()

    # ── 4. Мігрувати klines ───────────────────────────────────────────────────
    logger.info("--- klines: %d пар ---", len(all_klines))
    skipped_klines = 0
    for i, (sym, iv) in enumerate(all_klines, start=1):
        path = klines_path(out_dir, sym, iv)
        if path.exists() and not overwrite:
            logger.info("[%d/%d] klines %s %s: вже є (%s) — пропускаю",
                        i, len(all_klines), sym, iv, path.name)
            skipped_klines += 1
            continue

        logger.info("[%d/%d] klines %s %s: читаю з Postgres...", i, len(all_klines), sym, iv)
        t0 = time.monotonic()
        df = pg.load_klines(sym, iv)
        if df is None or df.empty:
            logger.warning("  -> порожній результат, пропускаю")
            continue

        logger.info("  -> %s рядків, зберігаю в %s...", _fmt_rows(len(df)), path.name)
        save_klines(path, df)
        elapsed = time.monotonic() - t0
        logger.info("  OK збережено за %.1f с", elapsed)
        stats[f"klines:{sym}:{iv}"] = len(df)

    if skipped_klines:
        logger.info(
            "Пропущено (вже є): %d klines-файлів. Використай --overwrite для перезапису.",
            skipped_klines,
        )

    # ── 5. Мігрувати aggTrades ────────────────────────────────────────────────
    if not skip_trades and all_symbols:
        logger.info("--- aggTrades: %d символів ---", len(all_symbols))
        for i, sym in enumerate(all_symbols, start=1):
            path = trades_path(out_dir, sym)
            if path.exists() and not overwrite:
                logger.info("[%d/%d] aggTrades %s: вже є (%s) — пропускаю",
                            i, len(all_symbols), sym, path.name)
                continue

            logger.info("[%d/%d] aggTrades %s: читаю з Postgres...", i, len(all_symbols), sym)
            t0 = time.monotonic()
            df = pg.load_trades(sym)
            if df is None or df.empty:
                logger.warning("  -> порожній результат, пропускаю")
                continue

            logger.info("  -> %s рядків, зберігаю в %s...", _fmt_rows(len(df)), path.name)
            save_trades(path, df)
            elapsed = time.monotonic() - t0
            logger.info("  OK збережено за %.1f с", elapsed)
            stats[f"trades:{sym}"] = len(df)
    elif skip_trades:
        logger.info("--- aggTrades: пропущено (--skip-trades) ---")

    # ── 6. Мігрувати funding ──────────────────────────────────────────────────
    if not skip_funding and all_symbols:
        logger.info("--- funding: %d символів ---", len(all_symbols))
        for i, sym in enumerate(all_symbols, start=1):
            path = funding_path(out_dir, sym)
            if path.exists() and not overwrite:
                logger.info("[%d/%d] funding %s: вже є (%s) — пропускаю",
                            i, len(all_symbols), sym, path.name)
                continue

            logger.info("[%d/%d] funding %s: читаю з Postgres...", i, len(all_symbols), sym)
            t0 = time.monotonic()
            df = pg.load_funding(sym)
            if df is None or df.empty:
                logger.warning("  -> порожній результат, пропускаю")
                continue

            logger.info("  -> %s рядків, зберігаю в %s...", _fmt_rows(len(df)), path.name)
            save_funding(path, df)
            elapsed = time.monotonic() - t0
            logger.info("  OK збережено за %.1f с", elapsed)
            stats[f"funding:{sym}"] = len(df)
    elif skip_funding:
        logger.info("--- funding: пропущено (--skip-funding) ---")

    # ── 7. Підсумок ───────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t0_total
    total_rows = sum(stats.values())
    logger.info("================================================")
    logger.info("Міграція завершена за %.1f с", total_elapsed)
    logger.info("Файлів збережено: %d", len(stats))
    logger.info("Рядків загалом:   %s", _fmt_rows(total_rows))
    logger.info("Директорія:       %s", out_dir)
    logger.info("================================================")

    if stats:
        logger.info("")
        logger.info("Наступний крок — перемкнути бекенд у .env:")
        logger.info("  DATA_BACKEND=parquet   (замість postgres)")
        logger.info("")
        logger.info("Після цього:")
        logger.info(
            "  uv run python -m scalper_hft.cli download "
            "--symbol BTCUSDT --interval 1m --days 7"
        )
        logger.info("  -> побачиш: 'кеш покрито — нічого докачувати'")

    return stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Міграція ринкових даних з PostgreSQL у Parquet-файли",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--symbol", default=None,
        help="Символи через кому (за замовч. — всі з Postgres). Приклад: BTCUSDT,ETHUSDT",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Перезаписати вже наявні Parquet-файли",
    )
    p.add_argument(
        "--skip-trades", action="store_true",
        help="Не мігрувати aggTrades (великі таблиці)",
    )
    p.add_argument(
        "--skip-funding", action="store_true",
        help="Не мігрувати funding rates",
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Директорія для Parquet (за замовч. DATA_DIR з .env)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Показати план міграції без запису файлів",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    symbols = (
        [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
        if args.symbol else None
    )
    data_dir = Path(args.data_dir) if args.data_dir else None

    migrate(
        symbols=symbols,
        overwrite=args.overwrite,
        skip_trades=args.skip_trades,
        skip_funding=args.skip_funding,
        data_dir=data_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
