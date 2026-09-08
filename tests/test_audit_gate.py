"""Тести hard-гейту overfitting-аудиту та fail-closed валідації барів (Фаза 1.4–1.5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scalper_hft.data.storage import save_klines
from scalper_hft.live.audit_gate import (
    audit_gate_check,
    audit_gate_check_pair,
    require_audit_pass,
    require_pair_audit_pass,
)
from scalper_hft.validation.verdict_store import latest_verdict, record_pair_verdict, record_verdict


@pytest.fixture()
def verdicts_path(tmp_path: Path) -> Path:
    return tmp_path / "audit_verdicts.jsonl"


def _pass(path: Path, symbol: str = "BTCUSDT", age_days: int = 0) -> None:
    record_verdict(
        "pairs_arb",
        symbol,
        "1h",
        "PASS",
        "",
        path=path,
        now=datetime.now(UTC) - timedelta(days=age_days),
    )


class TestVerdictStore:
    def test_record_and_latest(self, verdicts_path: Path) -> None:
        _pass(verdicts_path)
        record_verdict("pairs_arb", "BTCUSDT", "1h", "FAIL", "DSR=0.50≤0.95", path=verdicts_path)
        v = latest_verdict("pairs_arb", "BTCUSDT", "1h", path=verdicts_path)
        assert v is not None and v["label"] == "FAIL"

    def test_latest_scoped_by_cell(self, verdicts_path: Path) -> None:
        _pass(verdicts_path, symbol="BTCUSDT")
        assert latest_verdict("pairs_arb", "ETHUSDT", "1h", path=verdicts_path) is None
        assert latest_verdict("mean_reversion", "BTCUSDT", "1h", path=verdicts_path) is None
        assert latest_verdict("pairs_arb", "BTCUSDT", "4h", path=verdicts_path) is None

    def test_corrupt_lines_skipped(self, verdicts_path: Path) -> None:
        verdicts_path.write_text("{broken json\n", encoding="utf-8")
        _pass(verdicts_path)
        assert latest_verdict("pairs_arb", "BTCUSDT", "1h", path=verdicts_path) is not None


class TestAuditGate:
    def test_no_verdict_fails_closed(self, verdicts_path: Path) -> None:
        ok, msg = audit_gate_check("pairs_arb", "BTCUSDT", "1h", path=verdicts_path)
        assert not ok and "немає записаного аудиту" in msg

    def test_fail_verdict_blocks(self, verdicts_path: Path) -> None:
        record_verdict("pairs_arb", "BTCUSDT", "1h", "FAIL", "DSR низький", path=verdicts_path)
        ok, msg = audit_gate_check("pairs_arb", "BTCUSDT", "1h", path=verdicts_path)
        assert not ok and "FAIL" in msg

    def test_stale_pass_blocks(self, verdicts_path: Path) -> None:
        _pass(verdicts_path, age_days=60)
        ok, msg = audit_gate_check("pairs_arb", "BTCUSDT", "1h", path=verdicts_path)
        assert not ok and "протух" in msg

    def test_fresh_pass_ok(self, verdicts_path: Path) -> None:
        _pass(verdicts_path, age_days=3)
        ok, _ = audit_gate_check("pairs_arb", "BTCUSDT", "1h", path=verdicts_path)
        assert ok

    def test_require_all_symbols(self, verdicts_path: Path) -> None:
        _pass(verdicts_path, symbol="BTCUSDT")
        with pytest.raises(RuntimeError, match="ETHUSDT"):
            require_audit_pass("pairs_arb", ["BTCUSDT", "ETHUSDT"], "1h", path=verdicts_path)
        _pass(verdicts_path, symbol="ETHUSDT")
        require_audit_pass("pairs_arb", ["BTCUSDT", "ETHUSDT"], "1h", path=verdicts_path)


class TestPairAuditGate:
    def test_pair_pass_without_leg_symbols(self, verdicts_path: Path) -> None:
        record_pair_verdict("pairs_arb", "LINKUSDT", "BTCUSDT", "1h", "PASS", path=verdicts_path)
        ok, _ = audit_gate_check_pair("pairs_arb", "LINKUSDT", "BTCUSDT", "1h", path=verdicts_path)
        assert ok
        ok_leg, _ = audit_gate_check("pairs_arb", "BTCUSDT", "1h", path=verdicts_path)
        assert not ok_leg
        require_pair_audit_pass("pairs_arb", "LINKUSDT", "BTCUSDT", "1h", path=verdicts_path)

    def test_missing_pair_verdict_blocks_paper(self, verdicts_path: Path) -> None:
        from scalper_hft.live.pairs_runner import PairsPaperRunner

        with pytest.raises(RuntimeError, match="гейт пари"):
            PairsPaperRunner("AAA", "BBB", require_audit=True, audit_path=verdicts_path)

    def test_stale_pair_pass_blocks(self, verdicts_path: Path) -> None:
        record_pair_verdict(
            "pairs_arb",
            "LINKUSDT",
            "BTCUSDT",
            "1h",
            "PASS",
            path=verdicts_path,
            now=datetime.now(UTC) - timedelta(days=60),
        )
        with pytest.raises(RuntimeError, match="протух"):
            require_pair_audit_pass("pairs_arb", "LINKUSDT", "BTCUSDT", "1h", path=verdicts_path)


def _good_bars(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    c = np.full(n, 100.0)
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)


class TestStrictSaveKlines:
    def test_ohlc_violation_raises(self, tmp_path: Path) -> None:
        df = _good_bars()
        df.loc[df.index[3], "high"] = 90.0  # high < close — порушення
        with pytest.raises(ValueError, match="Відмова у збереженні"):
            save_klines(tmp_path / "BTCUSDT_1m_klines.parquet", df)
        assert not (tmp_path / "BTCUSDT_1m_klines.parquet").exists()

    def test_duplicates_raise(self, tmp_path: Path) -> None:
        df = _good_bars()
        df = pd.concat([df, df.iloc[[2]]]).sort_index()
        with pytest.raises(ValueError):
            save_klines(tmp_path / "x.parquet", df)

    def test_future_bars_raise(self, tmp_path: Path) -> None:
        idx = pd.date_range("2999-01-01", periods=10, freq="1min")
        df = _good_bars()
        df.index = idx
        with pytest.raises(ValueError):
            save_klines(tmp_path / "x.parquet", df)

    def test_good_bars_saved(self, tmp_path: Path) -> None:
        p = tmp_path / "BTCUSDT_1m_klines.parquet"
        save_klines(p, _good_bars())
        assert p.exists()

    def test_strict_false_warns_but_saves(self, tmp_path: Path) -> None:
        df = _good_bars()
        df.loc[df.index[3], "high"] = 90.0
        p = tmp_path / "x.parquet"
        save_klines(p, df, strict=False)
        assert p.exists()
