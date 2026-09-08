"""Тести exchange-side emergency flatten (Фаза 3.3): у live-режимі API kill
має скасовувати ордери і закривати позиції НА БІРЖІ, а не лише в SQLite."""

from __future__ import annotations

import pytest
from scalper_hft.api.server import _flatten_exchange_positions
from scalper_hft.config import Settings


class _FakeClient:
    """Записує дії flatten (без мережі)."""

    def __init__(self, *a, **kw) -> None:
        self.canceled_all = False
        self.closed: list[tuple[str, str, float]] = []
        self.fail_positions = False

    def cancel_all_orders(self, symbol=None) -> list:
        self.canceled_all = True
        return []

    def fetch_positions(self) -> list[dict]:
        if self.fail_positions:
            raise RuntimeError("exchange down")
        return [
            {"symbol": "BTCUSDT", "side": "long", "contracts": 0.5},
            {"symbol": "ETHUSDT", "side": "short", "contracts": 2.0},
        ]

    def create_order(self, symbol, order_type, side, amount, price=None, params=None, **kw) -> dict:
        assert order_type == "market"
        assert (params or {}).get("reduceOnly") is True
        self.closed.append((symbol, side, float(amount)))
        return {"id": "x", "status": "filled"}


def _live_settings() -> Settings:
    return Settings(binance_api_key="k", binance_api_secret="s", dry_run=False)


def test_paper_mode_no_exchange_calls(monkeypatch) -> None:
    """DRY_RUN=true → flatten лише локальний, біржа не чіпається."""
    monkeypatch.setattr("scalper_hft.api.server.get_settings", lambda: Settings(dry_run=True))
    actions, errors = _flatten_exchange_positions()
    assert actions == [] and errors == []


def test_live_flatten_cancels_and_closes(monkeypatch) -> None:
    """Live: cancel-all + reduce-only market close кожної позиції."""
    monkeypatch.setattr("scalper_hft.api.server.get_settings", _live_settings)
    client = _FakeClient()
    monkeypatch.setattr("scalper_hft.data.client.ExchangeClient", lambda *a, **kw: client)
    actions, errors = _flatten_exchange_positions()
    assert errors == []
    assert client.canceled_all
    # long → sell, short → buy
    assert ("BTCUSDT", "sell", 0.5) in client.closed
    assert ("ETHUSDT", "buy", 2.0) in client.closed
    assert len(actions) == 3  # cancel_all + 2 closes


def test_live_flatten_no_credentials_fails_closed(monkeypatch) -> None:
    """Live без ключів → RuntimeError (fail-closed), не тихий no-op."""
    monkeypatch.setattr(
        "scalper_hft.api.server.get_settings",
        lambda: Settings(binance_api_key="", binance_api_secret="", dry_run=False),
    )
    with pytest.raises(RuntimeError):
        _flatten_exchange_positions()


def test_live_flatten_reports_errors(monkeypatch) -> None:
    """Збій fetch_positions → помилка в списку, а не тихий success."""
    monkeypatch.setattr("scalper_hft.api.server.get_settings", _live_settings)
    client = _FakeClient()
    client.fail_positions = True
    monkeypatch.setattr("scalper_hft.data.client.ExchangeClient", lambda *a, **kw: client)
    actions, errors = _flatten_exchange_positions()
    assert any("fetch_positions" in e for e in errors)
    assert client.canceled_all  # cancel-all таки виконано
