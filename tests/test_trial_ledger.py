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
    """n_trials = max(варіанти цього аудиту, накопичений журнал).

    Регрес 2026-09-12: раніше сюди йшов повний декартів добуток `param_space`
    × «магічну» 50, тобто n_trials ~2·10⁶ для supertrend, хоча аудит запускає
    ОДНУ дефолтну конфігурацію. DSR з такою множинністю вимагав річного Sharpe
    ~4.7 — гейт ставав нездоланним (FAIL у 100% клітинок).
    """
    from scalper_hft.validation.trial_ledger import effective_n_trials, record_trial

    p = tmp_path / "ledger.jsonl"
    # без журналу → рівно варіанти цього аудиту (без множників)
    n0 = effective_n_trials(None, param_combinations=10)
    assert n0 == 10
    # журнал порожній → те саме
    assert effective_n_trials(p, param_combinations=10) == 10
    # дописали 700 спроб для цієї стратегії → n_trials має зрости до 700
    record_trial(p, strategy="mean_reversion", symbol="BTCUSDT", purpose="audit_cell", n_trials=700)
    n2 = effective_n_trials(p, param_combinations=10, strategy="mean_reversion")
    assert n2 == 700
    # інша стратегія не бачить чужих спроб
    assert effective_n_trials(p, param_combinations=10, strategy="pairs_arb") == 10
    # мінімум 1 навіть при нулі варіантів
    assert effective_n_trials(None, param_combinations=0) == 1


def test_record_trial_none_path_noop() -> None:
    from scalper_hft.validation.trial_ledger import count_trials, record_trial

    record_trial(None, strategy="x", symbol="y", purpose="t")  # не падає
    assert count_trials(None) == 0


def test_empty_or_dot_path_is_disabled_not_cwd() -> None:
    """Path('') == Path('.'): без TRIAL_LEDGER_PATH config віддавав Path(''),
    і count_trials падав IsADirectoryError (відкриття '.' як файлу).
    Тепер ''/'.' = журнал вимкнено: no-op замість краху."""
    from scalper_hft.validation.trial_ledger import count_trials, effective_n_trials, record_trial

    for disabled in (Path(""), Path(".")):
        # не падає і не читає теку директорію
        assert count_trials(disabled) == 0
        record_trial(disabled, strategy="s", symbol="x", purpose="t")  # не падає, нічого не пише
        assert count_trials(disabled) == 0
        # effective_n_trials повертає варіанти цього аудиту (журнал вимкнено)
        n = effective_n_trials(disabled, param_combinations=10)
        assert n == 10
