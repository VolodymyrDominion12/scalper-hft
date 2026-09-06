"""Регресійні тести для CLI команд та цілісності даних сховища (Фаза 1)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from scalper_hft.cli import main
from scalper_hft.config import get_settings
from scalper_hft.live.store import PaperStore


def test_cli_exchange_flag_no_frozen_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Перевірка, що прапорець --exchange не викликає FrozenInstanceError."""
    import os
    from scalper_hft.config import set_settings

    orig_settings = get_settings()
    orig_env = dict(os.environ)
    try:
        monkeypatch.setattr("scalper_hft.cli.cmd_download", lambda args: None)
        main(["download", "--symbol", "BTCUSDT", "--days", "1", "--exchange", "binance-custom"])
        settings = get_settings()
        assert settings.exchange == "binance-custom"
    finally:
        set_settings(orig_settings)
        os.environ.clear()
        os.environ.update(orig_env)


def test_cli_download_funding_keyword_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Перевірка, що download --funding викликає download_funding з правильними kwargs."""
    called_args: dict[str, object] = {}

    def fake_download_funding(symbol: str, days: int, force: bool = False, retries: int | None = None, batch_delay: float | None = None) -> pd.DataFrame:
        called_args["symbol"] = symbol
        called_args["days"] = days
        return pd.DataFrame()

    def fake_download_klines(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("scalper_hft.data.downloader.download_funding", fake_download_funding)
    monkeypatch.setattr("scalper_hft.data.downloader.download_klines", fake_download_klines)

    # Не повинно падати з TypeError
    main(["download", "--symbol", "BTCUSDT", "--days", "5", "--funding"])
    assert called_args.get("symbol") == "BTCUSDT"
    assert called_args.get("days") == 5


def test_cli_regime_backtest_execution() -> None:
    """Перевірка, що regime-backtest працює без помилок відсутності BacktestEngine або slippage."""
    with patch("scalper_hft.cli._load_klines") as mock_load:
        # Генеруємо синтетичні klines для швидкого тесту
        import numpy as np

        idx = pd.date_range("2026-01-01", periods=100, freq="1h")
        close = pd.Series(100.0 + np.arange(100, dtype=float), index=idx, dtype=float)
        mock_df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000.0,
            },
            index=idx,
        )
        mock_load.return_value = mock_df

        # Запуск regime-backtest з двома базовими стратегіями
        main([
            "regime-backtest",
            "--symbol", "BTCUSDT",
            "--strategies", "mean_reversion,supertrend",
            "--days", "10",
            "--interval", "1h",
            "--blend-mode", "exp3",
            "--no-regime-table",
        ])


def test_paper_store_log_position_alignment() -> None:
    """Перевірка захисту від зсуву аргументів у PaperStore.log_position."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_store.sqlite"
        store = PaperStore(db_path)

        # 1. Позиційний виклик без явного ts (симуляція live_monitor.py)
        store.log_position("binance", "BTCUSDT", "long", 0.25, 64500.0, 64700.0, 42.50, "paper")

        # 2. Сучасний виклик через keyword arguments
        store.log_position(
            exchange="bybit",
            symbol="ETHUSDT",
            side="short",
            size=2.0,
            entry_price=3000.0,
            mark_price=2980.0,
            unrealized_pnl=40.0,
            mode="paper",
        )

        df = store.open_positions()
        assert len(df) == 2

        btc_row = df[df["symbol"] == "BTCUSDT"].iloc[0]
        assert btc_row["exchange"] == "binance"
        assert btc_row["side"] == "long"
        assert float(btc_row["size"]) == 0.25
        assert float(btc_row["entry_price"]) == 64500.0
        assert float(btc_row["mark_price"]) == 64700.0
        assert float(btc_row["unrealized_pnl"]) == 42.50
        assert btc_row["mode"] == "paper"

        eth_row = df[df["symbol"] == "ETHUSDT"].iloc[0]
        assert eth_row["exchange"] == "bybit"
        assert eth_row["side"] == "short"
        assert float(eth_row["size"]) == 2.0
        assert float(eth_row["entry_price"]) == 3000.0
        assert float(eth_row["mark_price"]) == 2980.0
        assert float(eth_row["unrealized_pnl"]) == 40.0
        assert eth_row["mode"] == "paper"


def test_jwt_secret_key_length() -> None:
    """Перевірка, що api_secret_key має довжину >= 32 байти (RFC 7518 HS256)."""
    settings = get_settings()
    assert len(settings.api_secret_key.encode("utf-8")) >= 32
