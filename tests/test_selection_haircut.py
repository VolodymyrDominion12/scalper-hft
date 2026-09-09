"""Тести Bailey–López de Prado selection haircut (1C) — чесний вибір переможця серед N кандидатів."""

from __future__ import annotations

import numpy as np
from scalper_hft.validation.deflated_sharpe import selection_haircut


def test_selection_haircut_single_candidate_no_correction() -> None:
    idx, sr0, deflated = selection_haircut([0.5])
    assert idx == 0
    assert sr0 == 0.0
    assert deflated == 0.5  # без корекції при N=1


def test_selection_haircut_many_candidates_shaves_max() -> None:
    # багато кандидатів з шумом навколо нуля → очікуваний максимум > 0
    rng = np.random.default_rng(1)
    sharpes = list(rng.normal(0.0, 0.3, 50))
    idx, sr0, deflated = selection_haircut(sharpes)
    assert idx == int(np.argmax(sharpes))
    assert sr0 > 0.0  # haircut позитивний
    assert deflated < max(sharpes)  # зістрижено
    assert deflated == max(sharpes) - sr0


def test_selection_haircut_empty() -> None:
    idx, sr0, deflated = selection_haircut([])
    assert idx == -1 and sr0 == 0.0 and deflated == 0.0


def test_selection_haircut_ignores_nan() -> None:
    idx, sr0, deflated = selection_haircut([float("nan"), 0.4, 0.6, float("nan")])
    assert idx == 1  # 0.6 — найбільший серед чистих
    assert deflated == 0.6 - sr0


def test_selection_haircut_more_candidates_larger_sr0() -> None:
    # фіксований розкид, але більше кандидатів → більший haircut (очікуваний max зростає)
    rng = np.random.default_rng(2)
    base = list(rng.normal(0.0, 0.3, 10))
    _, sr0_few, _ = selection_haircut(base, n_trials=10)
    _, sr0_many, _ = selection_haircut(base, n_trials=200)
    assert sr0_many > sr0_few
