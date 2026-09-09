"""Unit-тести порогів pairs_gate (WF positive frac, n_windows)."""

from __future__ import annotations

import pytest
from scalper_hft.validation.pairs_gate import (
    MIN_TRADES,
    PBO_MAX,
    WF_POS_FRAC_MIN,
    evaluate_pair_wf_gate,
)


@pytest.mark.parametrize(
    ("pos_frac", "n_windows", "expected"),
    [
        (0.65, 10, "PASS"),
        (0.55, 5, "PASS"),
        (0.54, 5, "FAIL"),
        (0.0, 0, "FAIL"),
    ],
)
def test_evaluate_pair_wf_gate_boundaries(pos_frac: float, n_windows: int, expected: str) -> None:
    label, reasons = evaluate_pair_wf_gate(pos_frac, n_windows)
    assert label == expected
    if expected == "FAIL":
        assert reasons
    else:
        assert not reasons


def test_evaluate_pair_wf_gate_zero_windows_reason() -> None:
    label, reasons = evaluate_pair_wf_gate(0.0, 0)
    assert label == "FAIL"
    assert any("WF вікон=0" in r for r in reasons)


def test_pairs_gate_constants_documented() -> None:
    assert WF_POS_FRAC_MIN == 0.55
    assert PBO_MAX == 0.50
    assert MIN_TRADES == 20
