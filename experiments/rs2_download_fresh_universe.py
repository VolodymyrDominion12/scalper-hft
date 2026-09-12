"""RS-2 (iter16) — завантаження СВІЖОГО універсуму (символи поза CORE_15) для holdout-тесту.

Навіщо: усі 15 символів CORE уже використані під попередні гіпотези (див.
docs/reports/oos_usage.md). Щоб перевірити політику RS-1/RS-2 без підглядання,
потрібні ІНШІ імена з власною історією. Дані — тільки LIVE (binanceusdm).

Запуск:
    UV_CACHE_DIR=$PWD/.uvcache uv run python experiments/rs2_download_fresh_universe.py --days 1200
"""

from __future__ import annotations

import argparse

FRESH = [
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT",
    "FILUSDT", "ETCUSDT", "TRXUSDT", "ALGOUSDT", "EOSUSDT",
    "RUNEUSDT", "SANDUSDT",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--symbols", default=",".join(FRESH))
    args = ap.parse_args()

    from scalper_hft.data.downloader import download_funding, download_klines

    ok, fail = [], []
    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        for interval in ("1d", "4h"):
            try:
                df = download_klines(symbol, interval, args.days)
                print(f"{symbol} {interval}: {len(df)} барів {df.index[0]} → {df.index[-1]}", flush=True)
                ok.append((symbol, interval))
            except Exception as exc:  # noqa: BLE001
                print(f"{symbol} {interval}: FAIL {type(exc).__name__} {exc}", flush=True)
                fail.append((symbol, interval))
        try:
            fr = download_funding(symbol, args.days, strict_coverage=False)
            print(f"{symbol} funding: {len(fr)} рядків", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{symbol} funding: FAIL {type(exc).__name__} {exc}", flush=True)
    print(f"\nГотово: ok={len(ok)} fail={len(fail)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
