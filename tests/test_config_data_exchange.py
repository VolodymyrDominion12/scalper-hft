"""Live-біржа для ринкових даних: не testnet, дефолт EXCHANGE=binance."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
from scalper_hft.config import Settings, get_settings, require_live_data_exchange
from scalper_hft.data.client import ExchangeClient


def test_exchange_default_is_live_binance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXCHANGE", raising=False)
    settings = Settings()
    assert settings.exchange == "binance"
    assert "testnet" not in settings.exchange.lower()


def test_data_exchange_default_is_live_usdm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_EXCHANGE", raising=False)
    settings = Settings()
    assert settings.data_exchange == "binanceusdm"
    assert "testnet" not in settings.data_exchange.lower()


def test_require_live_data_exchange_rejects_testnet() -> None:
    settings = replace(get_settings(), data_exchange="binance-testnet", allow_testnet_data=False)
    with pytest.raises(RuntimeError, match="синтетичну"):
        require_live_data_exchange(settings)


def test_require_live_data_exchange_rejects_override_testnet() -> None:
    settings = replace(get_settings(), data_exchange="binanceusdm", allow_testnet_data=False)
    with pytest.raises(RuntimeError, match="синтетичну"):
        require_live_data_exchange(settings, "binance-testnet")


def test_require_live_data_exchange_accepts_binance() -> None:
    settings = replace(get_settings(), data_exchange="binanceusdm", allow_testnet_data=False)
    assert require_live_data_exchange(settings) == "binanceusdm"
    assert require_live_data_exchange(settings, "binance") == "binance"


def test_exchange_client_defaults_to_live_binance() -> None:
    default = inspect.signature(ExchangeClient.__init__).parameters["exchange_id"].default
    assert default == "binance"
    assert "testnet" not in str(default).lower()
