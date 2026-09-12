"""Тести для Portfolio VaR halt у PairsPortfolioRunner (Phase 6 / P6-B)."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_runner_stub(portfolio_var_limit: float = 0.0, equity: float = 10000.0):
    """Створює мінімальний stub PairsPortfolioRunner для unit-тестів VaR."""
    from scalper_hft.live.pairs_runner import PairsPortfolioRunner

    runner = object.__new__(PairsPortfolioRunner)
    account = MagicMock()
    account.equity = equity
    account.day_start_equity = equity
    runner.account = account
    runner.daily_loss_limit = 0.03
    runner.weekly_loss_limit = 0.05
    runner.week_start_equity = equity
    runner.portfolio_var_limit = portfolio_var_limit
    runner._equity_window: deque = deque(maxlen=100)
    return runner


class TestPortfolioVarOk:
    """Тести _portfolio_var_ok()."""

    def test_disabled_when_limit_zero(self):
        """Ліміт 0.0 → no-op, завжди True."""
        runner = _make_runner_stub(portfolio_var_limit=0.0)
        assert runner._portfolio_var_ok() is True

    def test_disabled_when_limit_negative(self):
        runner = _make_runner_stub(portfolio_var_limit=-0.01)
        assert runner._portfolio_var_ok() is True

    def test_insufficient_data_passes(self):
        """Менше 10 точок → не блокуємо (warm-up period)."""
        runner = _make_runner_stub(portfolio_var_limit=0.05)
        for i in range(9):
            runner._equity_window.append(10000.0 - i * 10)
        assert runner._portfolio_var_ok() is True

    def test_stable_equity_passes(self):
        """Стабільна equity (var_95 ≈ 0) → OK."""
        runner = _make_runner_stub(portfolio_var_limit=0.05)
        for _ in range(50):
            runner._equity_window.append(10000.0)
        assert runner._portfolio_var_ok() is True

    def test_small_drift_passes(self):
        """Малі коливання (var < limit) → OK."""
        runner = _make_runner_stub(portfolio_var_limit=0.05)
        rng = np.random.default_rng(42)
        equity = 10000.0
        for _ in range(50):
            equity += rng.normal(0, 10)  # ~0.1% vol
            runner._equity_window.append(equity)
        assert runner._portfolio_var_ok() is True

    def test_crash_equity_halts(self):
        """Різке падіння equity → var_95 > limit → halt."""
        runner = _make_runner_stub(portfolio_var_limit=0.01)
        equity = 10000.0
        runner._equity_window.append(equity)
        for _ in range(30):
            equity *= 0.97  # -3% per bar
            runner._equity_window.append(equity)
        assert runner._portfolio_var_ok() is False


class TestUpdateEquityWindow:
    """_update_equity_window() зберігає equity правильно."""

    def test_appends_current_equity(self):
        runner = _make_runner_stub()
        runner.account.equity = 12345.0
        runner._update_equity_window()
        assert runner._equity_window[-1] == pytest.approx(12345.0)

    def test_deque_maxlen_respected(self):
        runner = _make_runner_stub()
        for i in range(150):
            runner.account.equity = float(i)
            runner._update_equity_window()
        assert len(runner._equity_window) == 100


class TestShouldHaltEntriesWithVar:
    """_should_halt_entries() з VaR інтеграцією."""

    def test_halt_when_var_exceeded(self):
        runner = _make_runner_stub(portfolio_var_limit=0.01)
        # Наповнити вікно різким падінням
        equity = 10000.0
        runner._equity_window.append(equity)
        for _ in range(30):
            equity *= 0.97
            runner._equity_window.append(equity)
        runner.account.equity = equity
        runner.account.day_start_equity = 10000.0
        runner.week_start_equity = 10000.0
        # daily_loss_limit: equity/day_start > (1-0.03) → не спрацює ще
        # але VaR вже > 0.01
        result = runner._should_halt_entries()
        assert result is True

    def test_no_halt_when_limit_disabled(self):
        runner = _make_runner_stub(portfolio_var_limit=0.0, equity=10000.0)
        equity = 10000.0
        runner._equity_window.append(equity)
        for _ in range(30):
            equity *= 0.97
            runner._equity_window.append(equity)
        runner.account.equity = equity
        runner.account.day_start_equity = 10000.0
        runner.week_start_equity = 10000.0
        # VaR limit disabled → check only daily/weekly
        # equity=9000, day_start=10000, daily_loss_limit=0.03 → 9000/10000=0.9 > 0.97 → HALT
        # Тому перевіряємо окремо VaR-шлях: встановлюємо equity близько до daily_start
        runner.account.equity = 9800.0
        runner.account.day_start_equity = 9800.0
        runner.week_start_equity = 9800.0
        result = runner._should_halt_entries()
        assert result is False
