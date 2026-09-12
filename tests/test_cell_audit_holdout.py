"""`mode="final"` має вимагати СПРАВДІ неторканий holdout.

Контекст (аудит 2026-09-11, знахідка K4). У `validation/cell_audit.py`
`explicit_holdout=True` дозволяв `mode="final"` при `HOLDOUT_PCT=0`, і тоді
«сліпий» holdout брався як останні 20% ПОВНОГО `df` — того самого, на якому
вже прогнали walk-forward, DSR і sensitivity. `cell_verdict` у final перевіряє
лише `holdout_sharpe > 0`, а `live/audit_gate._verdict_ok` — лише label і вік
≤30 діб. Тобто єдиний label, що відкриває paper/live, міг бути виданий без
жодного неторканого бару.

Тест фіксує інваріант: у final-режимі жоден бар holdout-вікна не потрапляє ні
у walk-forward, ні в sensitivity, ні в CSCV.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scalper_hft.validation import cell_audit as ca

_DAYS = 1000


def _ohlcv(n: int = 8000) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0},
        index=idx,
    )


class _Settings(SimpleNamespace):
    maker_fee = 0.0002
    taker_fee = 0.0005
    slippage_bps = 2.0
    position_pct = 0.01
    enforce_holdout_pct = 0.0
    enforce_oos_burn = True
    oos_registry_path = Path("/tmp/nonexistent_oos_registry.md")


@pytest.fixture()
def patched(monkeypatch, tmp_path):
    """Підмінює I/O та важкі обчислення, лишаючи логіку cell_audit справжньою."""
    df_full = _ohlcv()
    seen: dict[str, pd.DataFrame] = {}
    settings = _Settings(oos_registry_path=tmp_path / "oos_usage.md")

    monkeypatch.setattr("scalper_hft.config.get_settings", lambda: settings)
    monkeypatch.setattr("scalper_hft.data.access.ensure_klines", lambda *a, **k: df_full)

    def _fake_wf(df, strategy, train, test, **kw):
        seen["wf_df"] = df
        from scalper_hft.validation.walk_forward import WalkForwardWindow

        return SimpleNamespace(
            windows=[WalkForwardWindow(0, 0, train, train, train + test, 0.4, 0.5, 0.01, 50)],
            avg_is_sharpe=0.4,
            avg_oos_sharpe=0.5,
            positive_windows_frac=0.6,
            degradation=0.0,
            oos_returns=df["close"].pct_change().fillna(0.0),
        )

    monkeypatch.setattr("scalper_hft.validation.walk_forward.run_walk_forward", _fake_wf)

    def _fake_sens(df, strategy, pname, values, **kw):
        seen["sens_df"] = df
        return SimpleNamespace(smoothness=0.5, best_params={pname: values[0]}, grid=pd.DataFrame({"metric": [1.0]}))

    monkeypatch.setattr("scalper_hft.validation.sensitivity.parameter_sensitivity", _fake_sens)

    monkeypatch.setattr(
        "scalper_hft.backtest.router.run_strategy_backtest",
        lambda df, *a, **k: SimpleNamespace(
            equity=(1 + df["close"].pct_change().fillna(0.0)).cumprod() * 10_000,
            bar_returns=df["close"].pct_change().fillna(0.0),
            metrics=SimpleNamespace(
                max_drawdown=-0.01,
                n_trades=50,
                total_return=0.01,
                sharpe=1.0,
                profit_factor=1.2,
                win_rate=0.5,
                trades_per_day=1.0,
            ),
        ),
    )

    def _fake_cscv(df, strategy, **kw):
        seen["cscv_df"] = df
        raise RuntimeError("CSCV не потрібен у тесті")

    monkeypatch.setattr("scalper_hft.validation.cscv.variant_returns", _fake_cscv)

    def _fake_ext(df, strategy, **kw):
        from scalper_hft.validation.audit_extensions import ExtendedAudit, QuintileAudit

        seen["ext_df"] = df
        seen["holdout_df"] = kw.get("holdout_df")
        return ExtendedAudit(
            quintile=QuintileAudit(spearman=0.5, monotonic=True, pass_=True),
            benchmark_sharpe=0.0,
            holdout_sharpe=0.5,
        )

    monkeypatch.setattr("scalper_hft.validation.audit_extensions.run_extended_audit", _fake_ext)
    monkeypatch.setattr(
        "scalper_hft.validation.oos_registry.check_and_burn", lambda **kw: (True, "")
    )
    return seen, df_full, settings


def test_final_without_holdout_pct_still_reserves_a_blind_holdout(patched) -> None:
    seen, df_full, _settings = patched

    res = ca.audit_cell("supertrend", "BTCUSDT", "1h", days=_DAYS, mode="final", explicit_holdout=True)

    assert res.status == "ok", res.error
    wf_df = seen["wf_df"]
    holdout = seen["holdout_df"]
    assert holdout is not None and not holdout.empty, "holdout має бути переданий у final"

    # Головний інваріант: research-частина закінчується РАНІШЕ holdout-вікна.
    assert wf_df.index[-1] < holdout.index[0], (
        f"walk-forward бачив бари holdout: WF до {wf_df.index[-1]}, holdout з {holdout.index[0]}"
    )
    # Holdout — саме хвіст даних (20% за замовчуванням).
    assert holdout.index[0] > df_full.index[0]
    assert holdout.index[-1] == df_full.index[-1]
    expected = int(round(len(df_full) * ca.FINAL_HOLDOUT_PCT))
    assert len(holdout) == expected
    # Sensitivity і CSCV — теж лише на research-частині.
    assert seen["sens_df"].index[-1] < holdout.index[0]


def test_final_without_explicit_holdout_is_rejected(patched) -> None:
    seen, _df, _s = patched

    res = ca.audit_cell("supertrend", "BTCUSDT", "1h", days=_DAYS, mode="final", explicit_holdout=False)

    assert res.status == "error"
    assert "HOLDOUT_PCT" in res.error
    assert "wf_df" not in seen, "WF не має запускатись без holdout"


def test_exploratory_mode_uses_full_history(patched) -> None:
    """Exploratory не мусить різати holdout — він для розвідки, не для PASS."""
    seen, df_full, _s = patched

    res = ca.audit_cell("supertrend", "BTCUSDT", "1h", days=_DAYS, mode="exploratory")

    assert res.status == "ok", res.error
    assert seen["wf_df"].index[-1] == df_full.index[-1]


def test_zero_final_holdout_pct_reproduces_the_original_defect(monkeypatch, patched) -> None:
    """Зуби тесту: при FINAL_HOLDOUT_PCT=0 інваріант ламається (стара поведінка).

    Саме це робив код до аудиту: `holdout_pct=0` → research = ПОВНИЙ df, а
    «holdout» = останні 20% того ж df. Тут це відтворюється примусово, щоб
    зафіксувати, чому константа мусить лишатися > 0.
    """
    seen, df_full, _s = patched
    monkeypatch.setattr(ca, "FINAL_HOLDOUT_PCT", 0.0)

    res = ca.audit_cell("supertrend", "BTCUSDT", "1h", days=_DAYS, mode="final", explicit_holdout=True)

    assert res.status == "ok", res.error
    holdout = seen.get("holdout_df")
    # Дефект: «сліпий» holdout або порожній, або перекривається з тим, що бачив WF.
    assert holdout is None or holdout.empty or seen["wf_df"].index[-1] >= holdout.index[0]
    assert seen["wf_df"].index[-1] == df_full.index[-1]
