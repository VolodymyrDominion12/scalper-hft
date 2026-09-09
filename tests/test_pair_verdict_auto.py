"""Тести авто pair-вердикту з overfit для pairs-стратегій (1C).

Перевіряє, що _maybe_record_pair_verdict:
- для pairs-стратегії з --leg1/--leg2 проганяє pairs WF і записує pair-вердикт;
- для non-pairs стратегії — не записує pair-вердикт;
- без --leg1/--leg2 — no-op.
"""

from __future__ import annotations

from types import SimpleNamespace


def _args(**kw):
    base = dict(
        strategy="pairs_arb",
        symbol="BTCUSDT",
        interval="1h",
        days=90,
        train=1500,
        test=500,
        trials=50,
        use_kalman=False,
        param_dict={},
        leg1=None,
        leg2=None,
        maker=True,
        position_pct=None,
        base=None,
        derive=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _patch_common(monkeypatch, wf_result, recorded) -> None:
    monkeypatch.setattr("scalper_hft.cli._load_klines", lambda *a, **k: None)
    monkeypatch.setattr("scalper_hft.backtest.pairs.run_pairs_walk_forward", lambda *a, **k: wf_result)
    monkeypatch.setattr(
        "scalper_hft.validation.verdict_store.record_pair_verdict",
        lambda strategy, l1, l2, interval, label, reasons="", **kw: recorded.update(
            strategy=strategy, l1=l1, l2=l2, interval=interval, label=label, reasons=reasons
        ),
    )
    monkeypatch.setattr(
        "scalper_hft.config.get_settings",
        lambda: SimpleNamespace(maker_fee=0.0002, taker_fee=0.0005, slippage_frac=0.0002),
    )
    monkeypatch.setattr("scalper_hft.data.downloader.download_funding", lambda *a, **k: None)


def test_pair_verdict_recorded_for_pairs(monkeypatch) -> None:
    from scalper_hft.cli.research_audit import _maybe_record_pair_verdict

    recorded: dict = {}
    _patch_common(monkeypatch, {"positive_windows": 0.7, "n_windows": 10, "avg_oos_sharpe": 0.01}, recorded)
    _maybe_record_pair_verdict(_args(leg1="XRPUSDT", leg2="BTCUSDT"))
    assert recorded["strategy"] == "pairs_arb"
    assert recorded["l1"] == "XRPUSDT" and recorded["l2"] == "BTCUSDT"
    assert recorded["label"] == "PASS"


def test_pair_verdict_fail_on_low_positive(monkeypatch) -> None:
    from scalper_hft.cli.research_audit import _maybe_record_pair_verdict

    recorded: dict = {}
    _patch_common(monkeypatch, {"positive_windows": 0.3, "n_windows": 10, "avg_oos_sharpe": 0.0}, recorded)
    _maybe_record_pair_verdict(_args(leg1="XRPUSDT", leg2="BTCUSDT"))
    assert recorded["label"] == "FAIL"
    assert "positive" in recorded["reasons"]


def test_pair_verdict_noop_without_legs(monkeypatch) -> None:
    from scalper_hft.cli.research_audit import _maybe_record_pair_verdict

    # без leg1/leg2 — не падає, нічого не робить
    _maybe_record_pair_verdict(_args(leg1=None, leg2=None))


def test_pair_verdict_skipped_for_directional(monkeypatch) -> None:
    from scalper_hft.cli.research_audit import _maybe_record_pair_verdict

    called = {"n": 0}

    def _wf(*a, **k):
        called["n"] += 1
        return {"positive_windows": 0.7, "n_windows": 10, "avg_oos_sharpe": 0.01}

    monkeypatch.setattr("scalper_hft.cli._load_klines", lambda *a, **k: None)
    monkeypatch.setattr("scalper_hft.backtest.pairs.run_pairs_walk_forward", _wf)
    monkeypatch.setattr(
        "scalper_hft.config.get_settings",
        lambda: SimpleNamespace(maker_fee=0.0002, taker_fee=0.0005, slippage_frac=0.0002),
    )
    monkeypatch.setattr("scalper_hft.data.downloader.download_funding", lambda *a, **k: None)
    # mean_reversion — не pairs, тому pair-WF не запускається
    _maybe_record_pair_verdict(_args(strategy="mean_reversion", leg1="XRPUSDT", leg2="BTCUSDT"))
    assert called["n"] == 0
