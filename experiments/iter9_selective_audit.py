"""iter9 — вибірковий аудит комірок (strategy × symbol × interval) після скрину.

Навіщо: скрин `sweep --mode backtest --days 1095` дає in-sample Sharpe для 15
символів × 3 таймфрейми. Повний аудит (WF + DSR + sensitivity + CSCV/PBO +
quintile/time-decay/stress) — дорогий, тому запускаємо його ЛИШЕ на комірках,
які скрин визнав перспективними (той самий принцип, що «тільки вибіркові»).

Запуск (послідовно — ProcessPool у цьому середовищі недоступний):
    .venv/bin/python experiments/iter9_selective_audit.py --tag iter9_core
    .venv/bin/python experiments/iter9_selective_audit.py --tag iter9_core --cells-file results/iter9/cells.json

Результат: results/iter9/audit_<tag>.csv (по рядку на комірку) + .jsonl (повний
CellAudit.to_summary_dict) + прогріс журналу спроб (docs/reports/trial_ledger.jsonl).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "iter9"
DAYS = 1095

# Вибіркові комірки: обґрунтування — у docs/reports/iter9_strategy_selection.md.
# Ядро (core) — те, що скрин 1095д визнав перспективним:
#   * cross_momentum на 1d — mean Sharpe +0.25, 10/15 символів > 0, ~96 угод
#     (єдина сім'я з edge І достатньою вибіркою на найдешевшому ТФ);
#   * cross_momentum на 4h — овертрейдинг (550 угод), але 2 символи сильно > 0;
#   * supertrend на 4h — єдиний ТФ, де трендовий механізм у нулі (топ SOL/BTC/BNB);
#   * supertrend/stoch_rsi на 1h — найкращі 1h-клітинки (перевірка, що 1h
#     провалюється саме на витратах, а не через брак сигналу).
CORE_CELLS: list[tuple[str, str, str]] = [
    # momentum × 1d (головний кандидат)
    ("cross_momentum", "DOGEUSDT", "1d"),
    ("cross_momentum", "ADAUSDT", "1d"),
    ("cross_momentum", "AVAXUSDT", "1d"),
    ("cross_momentum", "NEARUSDT", "1d"),
    ("cross_momentum", "DOTUSDT", "1d"),
    ("cross_momentum", "ATOMUSDT", "1d"),
    ("cross_momentum", "SOLUSDT", "1d"),
    ("cross_momentum", "XRPUSDT", "1d"),
    # momentum × 4h (fee-sensitive контроль)
    ("cross_momentum", "DOGEUSDT", "4h"),
    ("cross_momentum", "ADAUSDT", "4h"),
    # trend × 4h
    ("supertrend", "SOLUSDT", "4h"),
    ("supertrend", "BTCUSDT", "4h"),
    ("supertrend", "BNBUSDT", "4h"),
    ("supertrend", "LINKUSDT", "4h"),
    # trend/oscillator × 1h (найкращі клітинки 1h — контроль гіпотези «витрати»)
    ("supertrend", "ETHUSDT", "1h"),
    ("supertrend", "BNBUSDT", "1h"),
    ("stoch_rsi", "ETHUSDT", "4h"),
]


def _load_cells(cells_file: Path | None) -> list[tuple[str, str, str]]:
    if cells_file is None:
        return CORE_CELLS
    payload = json.loads(cells_file.read_text(encoding="utf-8"))
    return [(str(c["strategy"]), str(c["symbol"]), str(c["interval"])) for c in payload]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="iter9_core")
    ap.add_argument("--cells-file", type=Path, default=None)
    ap.add_argument("--no-cscv", action="store_true", help="без CSCV/PBO (швидко, exploratory)")
    ap.add_argument("--days", type=int, default=DAYS)
    args = ap.parse_args()

    cells = _load_cells(args.cells_file)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"audit_{args.tag}.csv"
    jsonl_path = OUT_DIR / f"audit_{args.tag}.jsonl"

    done: set[tuple[str, str, str]] = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = {(str(r.strategy), str(r.symbol), str(r.interval)) for r in prev.itertuples(index=False)}
        print(f"resume: {len(done)} комірок вже в {csv_path.name}")

    from scalper_hft.validation.cell_audit import audit_cell

    rows: list[dict] = []
    for i, (strategy, symbol, interval) in enumerate(cells, 1):
        if (strategy, symbol, interval) in done:
            continue
        t0 = time.time()
        audit = audit_cell(
            strategy,
            symbol,
            interval,
            args.days,
            with_cscv=not args.no_cscv,
        )
        summary = audit.to_summary_dict()
        rows.append(summary)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
        # інкрементальний запис: довгий прогін не втрачається при збої
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(
            f"[{i}/{len(cells)}] {strategy:<15s} {symbol:<10s} {interval:<3s} "
            f"status={summary.get('status')} oos={summary.get('avg_oos_sharpe')} "
            f"dsr={summary.get('dsr')} n_tr={summary.get('n_trials_dsr')} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    if rows:
        df = pd.read_csv(csv_path)
        print(f"\nЗбережено: {csv_path} ({len(df)} рядків)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
