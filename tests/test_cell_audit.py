"""Аудит комірки: пороги PASS/FAIL, train/test за ТФ, roundtrip JSON."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from scalper_hft.app_pages._common import (
    RESEARCH_BT_PREFILL,
    apply_research_audit_prefill,
    apply_research_bt_prefill,
    combo_prefill,
)
from scalper_hft.validation.cell_audit import (
    CellAudit,
    cell_verdict,
    default_purge_embargo,
    default_train_test,
    min_trades_for,
    resolve_wf_windows,
)


def test_default_purge_embargo_nonzero_afml() -> None:
    """AFML-дефолт: max(1, 1% OOS-вікна) — ненульовий прогін/ембарго."""
    assert default_purge_embargo(200) == (2, 2)
    assert default_purge_embargo(500) == (5, 5)
    # мінімум 1 бар навіть для крихітних вікон
    assert default_purge_embargo(50) == (1, 1)
    assert default_purge_embargo(10) == (1, 1)


def _ok(**over: object) -> CellAudit:
    base: dict[str, object] = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "strategy": "mean_reversion",
        "status": "ok",
        "avg_oos_sharpe": 0.50,
        "oos_pos_frac": 0.60,
        "dsr": 0.97,
        "smoothness": 0.40,
        "n_trades_oos": 40,
        "bt_n_trades": 55,
    }
    base.update(over)
    return CellAudit(**base)  # type: ignore[arg-type]


def test_default_train_test_known_and_fallback() -> None:
    assert default_train_test("1m") == (4000, 2000)
    assert default_train_test("4h") == (200, 100)
    assert default_train_test("1d") == (2000, 500)


def test_resolve_wf_windows_auto_and_override() -> None:
    assert resolve_wf_windows("4h") == (200, 100)
    assert resolve_wf_windows("1m") == (4000, 2000)
    assert resolve_wf_windows("4h", train_bars=2000, test_bars=500) == (2000, 500)
    with pytest.raises(ValueError):
        resolve_wf_windows("1h", train_bars=0, test_bars=10)


def test_min_trades_for_interval() -> None:
    assert min_trades_for("1m") == 100
    assert min_trades_for("1h") == 30
    assert min_trades_for("unknown") == 40


def test_cell_verdict_pass() -> None:
    label, why = cell_verdict(_ok())
    assert label == "PASS"
    assert why == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("avg_oos_sharpe", 0.3),
        ("oos_pos_frac", 0.49),
        ("dsr", 0.95),
        ("smoothness", 0.30),
        ("n_trades_oos", 29),
        ("pbo", 0.6),
    ],
)
def test_cell_verdict_fail_at_threshold(field: str, value: float) -> None:
    label, why = cell_verdict(_ok(**{field: value}))
    assert label == "FAIL"
    assert why


def test_cell_verdict_nan_and_missing() -> None:
    label, why = cell_verdict(_ok(avg_oos_sharpe=None, dsr=float("nan"), smoothness=None, n_trades_oos=None))
    assert label == "FAIL"
    assert "avg_oos_sharpe=nan" in why
    assert "DSR=nan" in why
    assert "smoothness=nan" in why
    assert "n_trades_oos=nan" in why


def test_cell_verdict_1m_requires_100_oos_trades() -> None:
    """Гейт на OOS-угоди: багато full-sample угод не рятує тонкий OOS."""
    fail, why = cell_verdict(_ok(interval="1m", n_trades_oos=99, bt_n_trades=500))
    assert fail == "FAIL"
    assert "n_trades_oos=99<100" in why
    passed, _ = cell_verdict(_ok(interval="1m", n_trades_oos=100))
    assert passed == "PASS"


def test_cell_verdict_pbo_checked_only_when_present() -> None:
    # pbo=None → CSCV не запускався → не гейтиться
    passed, _ = cell_verdict(_ok(pbo=None))
    assert passed == "PASS"
    fail, why = cell_verdict(_ok(pbo=0.75))
    assert fail == "FAIL"
    assert "PBO=0.75" in why


def test_cell_verdict_from_series() -> None:
    row = pd.Series(_ok().to_summary_dict())
    assert cell_verdict(row) == ("PASS", "")


def test_cell_audit_json_roundtrip() -> None:
    src = _ok(
        windows=({"window_idx": 0, "oos_sharpe": 0.4, "is_sharpe": 0.6, "oos_return": 0.01, "n_trades": 3},),
        sensitivity_grid=({"lookback": 20.0, "metric": 1.1},),
        sens_param="lookback",
    )
    restored = CellAudit.from_mapping(src.to_json_dict())
    assert restored.symbol == src.symbol
    assert restored.avg_oos_sharpe == pytest.approx(0.50)
    assert restored.windows[0]["window_idx"] == 0
    assert restored.sens_param == "lookback"
    assert cell_verdict(restored) == ("PASS", "")


def test_combo_prefill_nan_days() -> None:
    out = combo_prefill(
        {"strategy": "supertrend", "symbol": "ETHUSDT", "interval": "15m", "days": math.nan},
        default_days=90,
    )
    assert out["days"] == 90
    assert out["strategy"] == "supertrend"


def test_apply_research_bt_prefill() -> None:
    state: dict[str, object] = {
        RESEARCH_BT_PREFILL: {"strategy": "stoch_rsi", "symbol": "SOLUSDT", "interval": "4h", "days": 120}
    }
    apply_research_bt_prefill(state)
    assert RESEARCH_BT_PREFILL not in state
    assert state["bt_strategy"] == "stoch_rsi"
    assert state["bt_symbol"] == "SOLUSDT"
    assert state["bt_interval_single"] == "4h"
    assert state["bt_days"] == 120


def test_apply_research_audit_prefill() -> None:
    state: dict[str, object] = {
        "research_audit_prefill": {
            "strategy": "hmm_reversion",
            "symbol": "BTCUSDT",
            "interval": "5m",
            "days": 30,
        }
    }
    apply_research_audit_prefill(state)
    assert state["rs_strategy"] == "hmm_reversion"
    assert state["rs_symbol"] == "BTCUSDT"
    assert state["rs_interval"] == "5m"
    assert state["rs_days"] == 30
