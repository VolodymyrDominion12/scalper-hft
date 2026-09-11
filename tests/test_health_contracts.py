"""Regression/contract tests for platform health baseline (CHANGE_PLAN_HEALTH)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

_RUN_BACKTEST_IMPORT = re.compile(r"from\s+scalper_hft\.backtest\.engine\s+import\s+.*\brun_backtest\b")

_ALLOWED_RUN_BACKTEST_MODULES = {
    _REPO_ROOT / "scalper_hft" / "backtest" / "engine.py",
    _REPO_ROOT / "scalper_hft" / "backtest" / "router.py",
    _REPO_ROOT / "scalper_hft" / "backtest" / "__init__.py",
}

_TEST_RUN_BACKTEST_ALLOWLIST = {
    _REPO_ROOT / "tests" / "test_engine_perf.py",
    _REPO_ROOT / "tests" / "test_engine_realism.py",
    _REPO_ROOT / "tests" / "test_capability_contract.py",
    _REPO_ROOT / "tests" / "test_improvements.py",
    _REPO_ROOT / "tests" / "test_lookahead_fixes.py",
    _REPO_ROOT / "tests" / "test_p0.py",
    _REPO_ROOT / "tests" / "test_paper_maker_parity.py",
    _REPO_ROOT / "tests" / "test_ported_strategies.py",
    _REPO_ROOT / "tests" / "test_review_fixes.py",
    _REPO_ROOT / "tests" / "test_sprint1.py",
    _REPO_ROOT / "tests" / "test_system.py",
    _REPO_ROOT / "tests" / "test_visualization.py",
}

_CCXT_IMPORT = re.compile(r"(?:^|\s)(?:import\s+ccxt|from\s+ccxt\s+import)")
_CCXT_PRODUCTION = re.compile(r"ccxt\.(binance|bybit|okx|kraken|kucoin|pro)\s*\(|ccxt\.pro\b|ccxt\.async_support")

_CCXT_TEST_ALLOWLIST = {
    _REPO_ROOT / "tests" / "test_exchange_registry.py",
}


def _bars_df(n: int = 120, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0},
        index=idx,
    )


def test_router_uses_event_engine_for_market_maker(monkeypatch) -> None:
    """market_maker must route through the event engine (not vector run_backtest)."""
    from scalper_hft.backtest.event_engine import EventBacktestResult
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.strategies import get_strategy

    df = _bars_df(200)
    strategy = get_strategy("market_maker")
    event_calls: list[str] = []

    def _fake_event_backtest(*args, **kwargs):
        event_calls.append("event")
        return EventBacktestResult(
            equity=pd.Series(10_000.0, index=df.index),
            positions=pd.Series(0.0, index=df.index),
            trades=pd.DataFrame(),
            metrics=None,
        )

    def _fake_vector_backtest(*args, **kwargs):
        raise AssertionError("market_maker must not use vector run_backtest")

    monkeypatch.setattr("scalper_hft.backtest.router.run_event_backtest", _fake_event_backtest)
    monkeypatch.setattr("scalper_hft.backtest.router.run_backtest", _fake_vector_backtest)

    result = run_strategy_backtest(df, strategy, apply_settings_risk=False, strict_data=False)
    assert event_calls == ["event"]
    assert isinstance(result, EventBacktestResult)


def test_live_entrypoints_require_audit_when_not_dry_run(monkeypatch) -> None:
    """Live (dry_run=False) paths must call audit gates before trading loops."""
    live_audit_calls: list[tuple] = []
    pair_calls: list[tuple] = []

    def _live_audit(*args, **kwargs) -> None:
        live_audit_calls.append((args, kwargs))

    def _pair_audit(*args, **kwargs) -> None:
        pair_calls.append((args, kwargs))

    monkeypatch.setattr("scalper_hft.live.audit_gate.require_live_audit_if_not_dry_run", _live_audit)
    monkeypatch.setattr("scalper_hft.live.audit_gate.require_pair_audit_pass", _pair_audit)

    import scalper_hft.config as cfg
    from scalper_hft.cli.paper import _check_directional_audit_gate

    monkeypatch.setattr(
        cfg,
        "get_settings",
        lambda: SimpleNamespace(dry_run=False, require_audit_pass=True, audit_max_age_days=30),
    )
    _check_directional_audit_gate(
        SimpleNamespace(strategy="mean_reversion", symbol="BTCUSDT", interval="1h"),
    )
    assert len(live_audit_calls) == 1

    from scalper_hft.config import Settings
    from scalper_hft.live.pairs_runner import PairsLiveRunner
    from tests.test_pairs_live import MockExchangeClient

    monkeypatch.setattr(
        "scalper_hft.live.pairs_runner.get_settings",
        lambda: Settings(
            binance_api_key="k",
            binance_api_secret="s",
            dry_run=False,
            audit_max_age_days=30,
        ),
    )
    PairsLiveRunner(
        "AAA",
        "BBB",
        client=MockExchangeClient(),
        restore=False,
        require_audit=True,
    )
    assert len(pair_calls) == 1


def _python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if p.is_file()]


def _imports_run_backtest(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "scalper_hft.backtest.engine":
            if any(alias.name == "run_backtest" for alias in node.names):
                return True
    return False


def test_no_direct_run_backtest_outside_allowed_modules() -> None:
    """Direct engine import is confined to router/engine and parity test allowlist."""
    violations: list[str] = []

    for path in _python_files(_REPO_ROOT / "scalper_hft"):
        if path in _ALLOWED_RUN_BACKTEST_MODULES:
            continue
        if _imports_run_backtest(path):
            violations.append(str(path.relative_to(_REPO_ROOT)))

    tests_dir = _REPO_ROOT / "tests"
    for path in _python_files(tests_dir):
        if path in _TEST_RUN_BACKTEST_ALLOWLIST:
            continue
        if _imports_run_backtest(path):
            violations.append(str(path.relative_to(_REPO_ROOT)))

    assert not violations, "run_backtest imported outside allowlist:\n  " + "\n  ".join(sorted(violations))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
    return mods


def _cost_model_constructor_lines(path: Path) -> list[int]:
    """Line numbers of CostModel(...) calls, excluding CostModel.from_settings."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "CostModel":
            lines.append(int(getattr(node, "lineno", 0)))
        elif isinstance(func, ast.Attribute) and func.attr == "CostModel":
            lines.append(int(getattr(node, "lineno", 0)))
    return lines


def test_use_cases_does_not_import_market_data_io() -> None:
    """W0-R6: application.run_backtest must not load klines/trades itself."""
    path = _REPO_ROOT / "scalper_hft" / "application" / "use_cases.py"
    mods = _imported_modules(path)
    forbidden = {"scalper_hft.data.downloader", "scalper_hft.data.access"}
    assert not (mods & forbidden), f"use_cases imports market I/O: {sorted(mods & forbidden)}"


def _getattr_string_arg_lines(path: Path, attr: str) -> list[int]:
    """Line numbers of getattr(..., \"attr\"[, default]) in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue
        key = node.args[1]
        if isinstance(key, ast.Constant) and key.value == attr:
            lines.append(int(node.lineno))
    return lines


def test_no_getattr_use_exit_ladders() -> None:
    """W0-R4: Settings.use_exit_ladders is a real field — no getattr fallback."""
    violations: list[str] = []
    for path in _python_files(_REPO_ROOT / "scalper_hft"):
        for lineno in _getattr_string_arg_lines(path, "use_exit_ladders"):
            rel = path.relative_to(_REPO_ROOT)
            violations.append(f"{rel}:{lineno}")
    assert not violations, "getattr(..., 'use_exit_ladders') leftover:\n  " + "\n  ".join(violations)


def test_cli_and_application_cost_model_from_settings() -> None:
    """W0-COST: CLI/application must not build a flat CostModel(...)."""
    violations: list[str] = []
    for root in (
        _REPO_ROOT / "scalper_hft" / "cli",
        _REPO_ROOT / "scalper_hft" / "application",
    ):
        for path in _python_files(root):
            for lineno in _cost_model_constructor_lines(path):
                rel = path.relative_to(_REPO_ROOT)
                violations.append(f"{rel}:{lineno}")
    assert not violations, "flat CostModel() in cli/application:\n  " + "\n  ".join(violations)


def test_unit_tests_never_import_ccxt_production() -> None:
    """Unit tests may use ccxt exception types, not production exchange clients."""
    violations: list[str] = []

    for path in _python_files(_REPO_ROOT / "tests"):
        if path in _CCXT_TEST_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if not _CCXT_IMPORT.search(text):
            continue
        if _CCXT_PRODUCTION.search(text):
            violations.append(str(path.relative_to(_REPO_ROOT)))

    assert not violations, "tests import production ccxt clients:\n  " + "\n  ".join(sorted(violations))
