"""Шаблон дослідницького скрипту (pre-register гіпотезу в docs/reports/hypothesis_*.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

MIN_TRADES_3Y = 30


def main() -> None:
    parser = argparse.ArgumentParser(description="Research experiment template")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()

    from scalper_hft.config import get_settings
    from scalper_hft.validation.cell_audit import audit_cell, cell_verdict

    audit = audit_cell(
        args.strategy,
        args.symbol,
        args.interval,
        args.days,
        mode="exploratory",
    )
    if audit.status != "ok":
        raise SystemExit(f"audit error: {audit.error}")
    if (audit.bt_n_trades or 0) < MIN_TRADES_3Y:
        raise SystemExit(f"degenerate: n_trades={audit.bt_n_trades} < {MIN_TRADES_3Y}")

    label, reasons = cell_verdict(audit, mode="exploratory")
    out = Path("results") / f"template_{args.strategy}_{args.symbol}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"verdict={label}\nreasons={reasons}\nsettings={get_settings().data_exchange}\n", encoding="utf-8")
    print(label, reasons)


if __name__ == "__main__":
    main()
