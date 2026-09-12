"""Тести для Factor Tearsheet & Alpha Decay Engine."""

import numpy as np
import pandas as pd
import pytest
from scalper_hft.research.factor_tearsheet import (
    compute_forward_returns,
    estimate_alpha_half_life,
    format_tearsheet_table,
    run_factor_tearsheet,
)


def test_forward_returns_no_lookahead() -> None:
    """Перевірка коректності розрахунку форвардних ретьорнів."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1h")
    prices = pd.Series(np.linspace(100.0, 200.0, 100), index=dates)

    fwd_dict = compute_forward_returns(prices, horizons=(1, 5))
    assert 1 in fwd_dict
    assert 5 in fwd_dict

    # Останній елемент на горизонті 1 має бути NaN через shift(-1)
    assert np.isnan(fwd_dict[1].iloc[-1])
    assert not np.isnan(fwd_dict[1].iloc[-2])

    # Останні 5 елементів на горизонті 5 мають бути NaN
    assert fwd_dict[5].iloc[-5:].isna().all()
    # Перший ретьорн (t=0 до t=1)
    expected_ret_1 = (prices.iloc[1] / prices.iloc[0]) - 1.0
    assert pytest.approx(fwd_dict[1].iloc[0], rel=1e-5) == expected_ret_1


def test_perfect_predictive_factor() -> None:
    """Тест з ідеальним лінійним фактором (IC має бути близьким до 1.0)."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    # Генеруємо випадкове блукання ціни
    returns = np.random.normal(0, 0.01, size=n)
    prices = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=dates)

    # Фактор дорівнює точно наступному руху ціни
    fwd_1 = prices.shift(-1) / prices - 1.0
    factor = fwd_1.copy()

    res = run_factor_tearsheet(factor=factor, prices=prices, horizons=(1, 2), quantiles=5)

    rep1 = res.reports[1]
    assert rep1.ic_spearman > 0.95
    assert rep1.ic_pearson > 0.95
    assert rep1.p_value < 0.001
    assert rep1.is_monotonic is True
    assert rep1.spread_top_minus_bottom > 0.0


def test_pure_noise_factor() -> None:
    """Тест з чистим шумом (IC має бути близьким до 0.0, не статистично значущим)."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    prices = pd.Series(100.0 + np.random.normal(0, 1.0, size=n).cumsum(), index=dates)
    noise_factor = pd.Series(np.random.normal(0, 1.0, size=n), index=dates)

    res = run_factor_tearsheet(factor=noise_factor, prices=prices, horizons=(1, 4))
    for rep in res.reports.values():
        assert abs(rep.ic_spearman) < 0.1
        assert rep.p_value > 0.01  # Не значущий


def test_alpha_half_life_estimation() -> None:
    """Тест моделювання експоненційного затухання альфи."""
    from scalper_hft.research.factor_tearsheet import FactorHorizonReport

    # Створюємо репорти з штучним згасанням IC: IC(h) = 0.5 * exp(-0.1 * h)
    reports = {}
    for h in [1, 2, 4, 8, 16]:
        ic = 0.5 * np.exp(-0.1 * h)
        reports[h] = FactorHorizonReport(
            horizon=h,
            count=500,
            ic_pearson=float(ic),
            ic_spearman=float(ic),
            t_stat=5.0,
            p_value=0.0001,
        )

    half_life = estimate_alpha_half_life(reports)
    assert half_life is not None
    # Теорія: hl = ln(2) / 0.1 = 6.93
    assert pytest.approx(half_life, rel=0.1) == 6.93


def test_tearsheet_table_formatting() -> None:
    """Перевірка генерації текстової таблиці."""
    dates = pd.date_range("2025-01-01", periods=50, freq="1h")
    prices = pd.Series(np.linspace(100.0, 110.0, 50), index=dates)
    factor = pd.Series(np.linspace(1.0, 5.0, 50), index=dates)

    res = run_factor_tearsheet(factor=factor, prices=prices, horizons=(1, 2))
    table = format_tearsheet_table(res)
    assert "=== Factor Tearsheet: signal ===" in table
    assert "Horizon" in table
    assert "Rank IC" in table


def test_cmd_factor_audit_invocation(monkeypatch, capsys) -> None:
    """Перевірка виклику CLI cmd_factor_audit."""
    import argparse

    from scalper_hft.cli.research_audit import cmd_factor_audit
    from scalper_hft.data.research import MarketDataBundle

    dates = pd.date_range("2025-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(101, 111, 100),
            "low": np.linspace(99, 109, 100),
            "close": np.linspace(100, 110, 100),
            "volume": np.ones(100) * 10.0,
        },
        index=dates,
    )
    bundle = MarketDataBundle(klines=df)
    monkeypatch.setattr("scalper_hft.data.research.load_research_data", lambda *args, **kwargs: bundle)

    args = argparse.Namespace(
        symbol="BTCUSDT",
        interval="1h",
        days=10,
        factor="momentum",
        lookback=5,
        horizons="1,2,5",
        quantiles=5,
        leg1=None,
        leg2=None,
    )
    cmd_factor_audit(args)
    captured = capsys.readouterr()
    assert "Factor Tearsheet: momentum(BTCUSDT, lb=5)" in captured.out
    assert "Rank IC" in captured.out
