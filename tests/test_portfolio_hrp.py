"""Тести для Hierarchical Risk Parity (HRP) портфельного розподілу капіталу."""

import numpy as np
import pytest
from scalper_hft.portfolio.hrp import (
    correlation_distance,
    hrp_weights,
)


def test_correlation_distance() -> None:
    """Перевірка метрики відстані на кореляційній матриці."""
    corr = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.5], [-1.0, 0.5, 1.0]])
    dist = correlation_distance(corr)
    assert dist[0, 0] == 0.0
    assert dist[1, 1] == 0.0
    # rho = -1 => dist = sqrt(0.5 * 2) = 1.0
    assert pytest.approx(dist[0, 2]) == 1.0
    # rho = 0 => dist = sqrt(0.5)
    assert pytest.approx(dist[0, 1]) == np.sqrt(0.5)


def test_hrp_weights_basic() -> None:
    """Базовий тест HRP з 4 активами."""
    np.random.seed(42)
    t = 200
    # Актив 0: низька волатильність (0.01)
    # Актив 1: середня волатильність (0.02)
    # Актив 2: висока волатильність (0.05)
    # Актив 3: дуже висока волатильність (0.10)
    r = np.column_stack(
        [
            np.random.normal(0, 0.01, size=t),
            np.random.normal(0, 0.02, size=t),
            np.random.normal(0, 0.05, size=t),
            np.random.normal(0, 0.10, size=t),
        ]
    )

    w = hrp_weights(r)
    assert len(w) == 4
    assert pytest.approx(w.sum()) == 1.0
    assert (w >= 0.0).all()

    # Актив з найнижчим ризиком має отримати більше ваги, ніж актив з найбільшим ризиком
    assert w[0] > w[3]


def test_hrp_singular_collinear_assets() -> None:
    """Тест на стійкість HRP до сингулярних/колінеарних активів (де MVO падає)."""
    np.random.seed(42)
    t = 100
    base_r = np.random.normal(0, 0.02, size=t)
    # Актив 0 і 1 практично ідентичні (rho ~ 1.0)
    # Актив 2 незалежний
    r = np.column_stack(
        [
            base_r,
            base_r + 1e-9 * np.random.normal(0, 1.0, size=t),
            np.random.normal(0, 0.02, size=t),
        ]
    )

    # Не повинно кидати помилку LinAlgError / Singular Matrix
    w = hrp_weights(r)
    assert len(w) == 3
    assert pytest.approx(w.sum()) == 1.0
    assert (w >= 0.0).all()


def test_hrp_dead_asset() -> None:
    """Тест на обробку активу без варіації (константний нуль)."""
    np.random.seed(42)
    t = 100
    r = np.column_stack(
        [
            np.random.normal(0, 0.02, size=t),
            np.zeros(t),  # dead asset
            np.random.normal(0, 0.03, size=t),
        ]
    )

    w = hrp_weights(r)
    assert len(w) == 3
    assert pytest.approx(w.sum()) == 1.0
    assert w[1] == 0.0  # Мертвий актив отримує нульову вагу
    assert w[0] > 0.0
    assert w[2] > 0.0
