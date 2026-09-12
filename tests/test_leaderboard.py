"""Тести агрегації лідерборду."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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


def test_pairs_single_symbol_is_not_paper_ready(tmp_path: Path) -> None:
    verdicts = tmp_path / "audit_verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "strategy": "pairs_arb",
                "symbol": "LINKUSDT",
                "interval": "1h",
                "status": "ok",
                "avg_oos_sharpe": 0.5,
                "oos_pos_frac": 0.8,
                "dsr": 0.99,
                "pbo": 0.1,
                "n_trades_oos": 40,
                "smoothness": 0.5,
                "quintile_pass": True,
                "time_decay_pass": True,
                "stress_pass": True,
                "label": "PASS",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = build_leaderboard(tmp_path)
    link_only = [r for r in rows if r.strategy == "pairs_arb" and r.symbol == "LINKUSDT"]
    assert link_only
    assert link_only[0].tier == "rejected"


def test_link_btc_aliases_deduped(tmp_path: Path) -> None:
    verdicts = tmp_path / "audit_verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "strategy": "pairs_arb",
                "symbol": "LINKUSDT/BTCUSDT",
                "interval": "1h",
                "status": "ok",
                "avg_oos_sharpe": 0.01,
                "oos_pos_frac": 0.65,
                "dsr": 0.99,
                "pbo": 0.0,
                "n_trades_oos": 40,
                "smoothness": 0.5,
                "quintile_pass": True,
                "time_decay_pass": True,
                "stress_pass": True,
                "label": "PASS",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = build_leaderboard(tmp_path)
    pairs = [r for r in rows if r.strategy == "pairs_arb"]
    assert len(pairs) == 1
    assert pairs[0].tier == "validated_pairs"


def test_post_hoc_top10_is_candidate_not_monitoring(tmp_path: Path) -> None:
    csv_dir = tmp_path / "iter10"
    csv_dir.mkdir()
    pd.DataFrame(
        [
            {
                "variant": "smooth3_top10",
                "port_sharpe": 1.36,
                "t_newey_west": 3.1,
                "mean_oos_wf": 0.27,
                "symbols_pos_frac": 0.9,
                "notes": "smooth=3, short=True, n_sym=10",
            }
        ]
    ).to_csv(csv_dir / "variants.csv", index=False)
    rows = build_leaderboard(tmp_path)
    top10 = [r for r in rows if "top10" in r.symbol.lower()]
    assert top10
    assert top10[0].tier == "candidate"


def test_construction_evidence_is_candidate_not_monitoring(tmp_path: Path) -> None:
    """Value-added бленд рукавів (portfolio-construction evidence) — не paper-gate:
    слабкий рукав не промотується в monitoring через комбінацію (iter13 H13-B)."""
    csv_dir = tmp_path / "iter13"
    csv_dir.mkdir()
    pd.DataFrame(
        [
            {
                "variant": "combined_1w1d_lb13",
                "port_sharpe": 1.51,
                "t_newey_west": 2.61,
                "mean_oos_wf": 0.001,
                "symbols_pos_frac": 0.9,
                "notes": "H13-B value-added 50/50 1w⊕1d; portfolio-construction evidence",
            }
        ]
    ).to_csv(csv_dir / "variants.csv", index=False)
    rows = build_leaderboard(tmp_path)
    comb = [r for r in rows if "combined" in r.symbol.lower()]
    assert comb
    assert comb[0].tier == "candidate"
