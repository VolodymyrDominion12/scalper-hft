"""Тести агрегації лідерборду."""

from __future__ import annotations

import json
from pathlib import Path

from scalper_hft.research.leaderboard import (
    build_leaderboard,
    render_leaderboard_markdown,
)


def test_build_leaderboard_includes_validated_pairs(tmp_path: Path) -> None:
    rows = build_leaderboard(tmp_path)
    pairs = [r for r in rows if r.strategy == "pairs_arb"]
    assert pairs
    assert pairs[0].tier == "validated_pairs"


def test_build_leaderboard_reads_audit_jsonl(tmp_path: Path) -> None:
    verdicts = tmp_path / "audit_verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "strategy": "ts_momentum",
                "symbol": "DOGEUSDT",
                "interval": "1d",
                "status": "ok",
                "avg_oos_sharpe": 0.35,
                "oos_pos_frac": 0.6,
                "dsr": 0.5,
                "pbo": 0.3,
                "n_trades_oos": 50,
                "smoothness": 0.4,
                "quintile_pass": True,
                "time_decay_pass": True,
                "stress_pass": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = build_leaderboard(tmp_path)
    doge = [r for r in rows if r.symbol == "DOGEUSDT" and r.strategy == "ts_momentum"]
    assert doge
    assert doge[0].avg_oos_sharpe == 0.35


def test_render_leaderboard_markdown(tmp_path: Path) -> None:
    md = render_leaderboard_markdown(tmp_path, top_n=5)
    assert "# Лідерборд" in md
    assert "pairs_arb" in md
