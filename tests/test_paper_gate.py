"""Тести для check_paper_gate() — Hard IS Paper Gate (Phase 6 / P6-A)."""

from __future__ import annotations

import pytest

from scalper_hft.validation.paper_audit import (
    PAPER_GATE_THRESHOLDS,
    PaperAudit,
    check_paper_gate,
    format_gate_result,
)
from scalper_hft.validation.forensics import ForensicsReport


def _dummy_forensics() -> ForensicsReport:
    from scalper_hft.validation.forensics import MfeMaeSummary
    return ForensicsReport(
        n_closed=10,
        n_winners=5,
        n_losers=5,
        winrate=0.5,
        total_profit=0.0,
        expectancy=0.0,
        max_losing_streak=2,
        max_winning_streak=2,
        mfe_mae=MfeMaeSummary(
            winners_mfe_median=0.0,
            mfe_threshold=0.0,
            losers_mfe_median=0.0,
            losers_mfe_mean=0.0,
            losers_mae_mean=0.0,
            roundtrip_exits=0,
            entry_failures=0,
            losers_with_mfe=0,
        ),
    )


def _make_is_report(blended_tca_bps: float, *, coverage_ok: bool = True, fill_rate: float = 0.80):
    from scalper_hft.live.is_report import ISReport
    return ISReport(
        n_fills=25,
        n_unfilled=5,
        median_is_bps=blended_tca_bps,
        maker_is_bps=blended_tca_bps,
        taker_is_bps=blended_tca_bps * 2,
        median_miss_bps=1.0,
        fill_rate=fill_rate,
        blended_tca_bps=blended_tca_bps,
        model_slippage_bps=2.0,
        coverage_ok=coverage_ok,
        mid_distinct=True,
    )


def _make_audit(
    *,
    tracking_error=0.01,
    paper_fill_rate=0.80,
    paper_max_dd=0.05,
    bt_max_dd=0.05,
    dd_gate_ok=True,
    blended_tca_bps=None,
    tca_coverage_ok=True,
) -> PaperAudit:
    is_report = None
    if blended_tca_bps is not None:
        is_report = _make_is_report(blended_tca_bps, coverage_ok=tca_coverage_ok, fill_rate=paper_fill_rate)
    return PaperAudit(
        n_bars=100,
        paper_return=0.05,
        bt_return=0.04,
        tracking_error=tracking_error,
        paper_max_dd=paper_max_dd,
        bt_max_dd=bt_max_dd,
        dd_gate_ok=dd_gate_ok,
        paper_fill_rate=paper_fill_rate,
        bt_fill_rate=0.82,
        fill_gap=-0.02,
        forensics=_dummy_forensics(),
        is_report=is_report,
    )


def test_gate_all_ok():
    audit = _make_audit(blended_tca_bps=1.5)
    passed, failures = check_paper_gate(audit)
    assert passed is True
    assert failures == []


def test_gate_no_is_report_passes():
    """Відсутній ISReport не є провалом (ще не накопичили fills)."""
    audit = _make_audit(blended_tca_bps=None)
    passed, failures = check_paper_gate(audit)
    assert passed is True


def test_gate_tracking_error_none_not_fail():
    audit = _make_audit(tracking_error=None, blended_tca_bps=1.0)
    passed, failures = check_paper_gate(audit)
    assert passed is True


def test_gate_tracking_error_too_high():
    audit = _make_audit(tracking_error=0.05, blended_tca_bps=1.0)
    passed, failures = check_paper_gate(audit)
    assert passed is False
    assert any("tracking_error" in f for f in failures)


def test_gate_fill_rate_too_low():
    audit = _make_audit(paper_fill_rate=0.50, blended_tca_bps=1.0)
    passed, failures = check_paper_gate(audit)
    assert passed is False
    assert any("fill_rate" in f for f in failures)


def test_gate_dd_gate_failed():
    audit = _make_audit(dd_gate_ok=False, paper_max_dd=0.20, bt_max_dd=0.05, blended_tca_bps=1.0)
    passed, failures = check_paper_gate(audit)
    assert passed is False
    assert any("paper_max_dd" in f for f in failures)


def test_gate_blended_tca_too_high():
    audit = _make_audit(blended_tca_bps=5.5)
    passed, failures = check_paper_gate(audit)
    assert passed is False
    assert any("blended_tca_bps" in f for f in failures)


def test_gate_tca_no_coverage_skip():
    """coverage_ok=False → TCA не перевіряється (ненадійний IS)."""
    audit = _make_audit(blended_tca_bps=10.0, tca_coverage_ok=False)
    passed, failures = check_paper_gate(audit)
    assert not any("blended_tca_bps" in f for f in failures)


def test_gate_multiple_failures():
    audit = _make_audit(
        tracking_error=0.10,
        paper_fill_rate=0.40,
        blended_tca_bps=8.0,
    )
    passed, failures = check_paper_gate(audit)
    assert passed is False
    assert len(failures) >= 3


def test_gate_custom_tca_threshold_strict():
    audit = _make_audit(blended_tca_bps=2.5)
    passed, _ = check_paper_gate(audit, thresholds={**PAPER_GATE_THRESHOLDS, "blended_tca_bps_max": 2.0})
    assert passed is False


def test_gate_custom_tca_threshold_lenient():
    audit = _make_audit(blended_tca_bps=2.5)
    passed, _ = check_paper_gate(audit, thresholds={**PAPER_GATE_THRESHOLDS, "blended_tca_bps_max": 5.0})
    assert passed is True


def test_format_gate_result_pass():
    msg = format_gate_result(True, [])
    assert "PASS" in msg and "✅" in msg


def test_format_gate_result_fail():
    failures = ["fill_rate=40% < 70%", "blended_tca_bps=5.5 > 3.0"]
    msg = format_gate_result(False, failures)
    assert "FAIL" in msg and "❌" in msg
    for f in failures:
        assert f in msg
