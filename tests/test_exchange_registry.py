import ccxt
import pytest
from scalper_hft.data.exchange_registry import ExchangeMeta, ExchangeRegistry


def test_registry_has_defaults() -> None:
    exchanges = ExchangeRegistry.list_supported()
    assert "binance" in exchanges
    assert "bybit" in exchanges
    assert "okx" in exchanges


def test_registry_get_existing() -> None:
    meta = ExchangeRegistry.get("binance")
    assert isinstance(meta, ExchangeMeta)
    assert meta.id == "binance"
    assert meta.ccxt_class == ccxt.binance
    assert meta.default_maker_fee == 0.0002
    assert meta.default_taker_fee == 0.0005


def test_registry_get_fallback() -> None:
    meta = ExchangeRegistry.get("kraken")
    assert meta.id == "kraken"
    assert meta.ccxt_class == ccxt.kraken


def test_registry_get_invalid() -> None:
    with pytest.raises(ValueError, match="Невідома біржа: unknown_exchange"):
        ExchangeRegistry.get("unknown_exchange")
