"""Тести append-only журналу спроб (trials) — замість «магічної» 50 для DSR."""

from __future__ import annotations

from pathlib import Path


def test_record_and_count_trials(tmp_path: Path) -> None:
    from scalper_hft.validation.trial_ledger import count_trials, record_trial

    p = tmp_path / "ledger.jsonl"
    record_trial(p, strategy="mean_reversion", symbol="BTCUSDT", purpose="audit_cell", n_trials=50, score=0.4)
    record_trial(p, strategy="mean_reversion", symbol="BTCUSDT", purpose="audit_cell", n_trials=50, score=0.5)
    record_trial(p, strategy="pairs_arb", symbol="BTCUSDT", purpose="audit_cell", n_trials=30)
    # сума n_trials для mean_reversion/BTCUSDT = 100
    assert count_trials(p, strategy="mean_reversion", symbol="BTCUSDT") == 100
    # глобально = 130
    assert count_trials(p) == 130
    # фільтр по purpose
    assert count_trials(p, purpose="audit_cell") == 130


def test_effective_n_trials_uses_ledger_floor(tmp_path: Path) -> None:
    from scalper_hft.validation.trial_ledger import effective_n_trials, record_trial

    p = tmp_path / "ledger.jsonl"
    # без журналу → оцінка combos×backtests_per_combo
    n0 = effective_n_trials(None, param_combinations=10, backtests_per_combo=50)
    assert n0 == 500
    # журнал порожній → те саме
    n1 = effective_n_trials(p, param_combinations=10, backtests_per_combo=50)
    assert n1 == 500
    # дописали 700 спроб для цієї комірки → n_trials має зрости до 700
    record_trial(p, strategy="mean_reversion", symbol="BTCUSDT", purpose="audit_cell", n_trials=700)
    n2 = effective_n_trials(
        p, param_combinations=10, backtests_per_combo=50, strategy="mean_reversion", symbol="BTCUSDT"
    )
    assert n2 == 700
    # інша комірка не впливає
    n3 = effective_n_trials(p, param_combinations=10, backtests_per_combo=50, strategy="pairs_arb", symbol="BTCUSDT")
    assert n3 == 500


def test_record_trial_none_path_noop() -> None:
    from scalper_hft.validation.trial_ledger import count_trials, record_trial

    record_trial(None, strategy="x", symbol="y", purpose="t")  # не падає
    assert count_trials(None) == 0
