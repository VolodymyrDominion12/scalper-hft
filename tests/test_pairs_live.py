"""Тести PairsLiveAdapter з мок-біржею (без мережі).

Викривлення (DEPLOY_PLAN Фаза 3):
  - реальні post_only ліміти на вході (ніколи market)
  - all-or-none + chase/unwind (resolve_legging)
  - reconcile на старті: порожній+порожній=ok; порожній+є ноги=гідратація; drift=KillSwitch
  - cancel_pending скасує реальні ордери
  - reduce_only на виході
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.pairs_live import PairsLiveAdapter
from scalper_hft.live.reconcile import KillSwitch
from scalper_hft.strategies.pairs_arb import PairsArb


@dataclass
class MockOrder:
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    post_only: bool
    reduce_only: bool
    client_order_id: str
    status: str = "open"  # open | filled | canceled
    filled_qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class MockExchangeClient:
    """Мок біржі: create_order/cancel_order/fetch_order/fetch_positions/fetch_klines."""

    orders: dict[str, MockOrder] = field(default_factory=dict)  # oid -> MockOrder
    positions: list[dict] = field(default_factory=list)
    klines: dict[str, list] = field(
        default_factory=lambda: {"AAA": [[1, 2, 3, 4, 100.0, 5]], "BBB": [[1, 2, 3, 4, 50.0, 5]]}
    )
    _next_oid: int = 1
    created_calls: list[dict] = field(default_factory=list)
    canceled_oids: list[str] = field(default_factory=list)

    def create_order(
        self, symbol, order_type, side, amount, price=None, params=None, post_only=False, client_order_id=None
    ):
        oid = f"ex{self._next_oid}"
        self._next_oid += 1
        reduce_only = bool((params or {}).get("reduceOnly", False))
        o = MockOrder(
            oid, symbol, side, float(amount), float(price or 0.0), post_only, reduce_only, client_order_id or oid
        )
        self.orders[oid] = o
        self.created_calls.append(
            {"symbol": symbol, "type": order_type, "side": side, "post_only": post_only, "reduce_only": reduce_only}
        )
        return {"id": oid, "status": "open", "filled": 0.0}

    def cancel_order(self, order_id, symbol):
        self.canceled_oids.append(order_id)
        if order_id in self.orders:
            self.orders[order_id].status = "canceled"
        return {"id": order_id, "status": "canceled"}

    def cancel_all_orders(self, symbol=None):
        for oid, o in list(self.orders.items()):
            if symbol is None or o.symbol == symbol:
                o.status = "canceled"
                self.canceled_oids.append(oid)
        return []

    def fetch_order(self, order_id, symbol):
        o = self.orders.get(order_id)
        if o is None:
            return {"status": "not_found", "filled": 0.0}
        return {"id": o.order_id, "status": o.status, "filled": o.filled_qty, "average": o.avg_price, "price": o.price}

    def fetch_positions(self, symbols=None):
        return list(self.positions)

    def fetch_klines(self, symbol, timeframe, since_ms, limit=1000):
        return self.klines.get(symbol, [[1, 2, 3, 4, 100.0, 5]])

    def fill_order(self, oid: str, avg_price: float | None = None) -> None:
        """Допоміжна: заповнити ордер (симуляція біржі)."""
        o = self.orders[oid]
        o.status = "filled"
        o.filled_qty = o.size
        o.avg_price = avg_price or o.price


def _make_adapter(client: MockExchangeClient | None = None, legging: str = "chase") -> PairsLiveAdapter:
    client = client or MockExchangeClient()
    acc = PaperAccount(10_000.0, taker_fee=0.0005, maker_fee=0.0002)
    adapter = PairsLiveAdapter(
        "AAA",
        "BBB",
        PairsArb(lookback=20),
        acc,
        client=client,
        wait_bars=1,
        is_maker=True,
        legging_mode=legging,
        coint_kill=False,
    )
    return adapter


def test_place_legs_on_quote() -> None:
    """Сигнал входу → реальні post_only ліміти обох ніг розміщені."""
    client = MockExchangeClient()
    adapter = _make_adapter(client)
    ts = pd.Timestamp("2025-01-01 00:00")
    # want=+1 → short leg1 / long leg2; _quote розміщує реальні ордери
    adapter._quote(ts, want=1, p1=100.0, p2=50.0)
    assert adapter._live_pending is not None
    assert len(client.orders) == 2
    # обидва post_only
    assert all(c["post_only"] for c in client.created_calls)
    # обидва limit (не market)
    assert all(c["type"] == "limit" for c in client.created_calls)
    # ноги різних сторін: want=+1 → sell AAA (short), buy BBB (long)
    sides = {c["symbol"]: c["side"] for c in client.created_calls}
    assert sides["AAA"] == "sell"
    assert sides["BBB"] == "buy"


def test_never_market_on_entry() -> None:
    """На вході НІКОЛИ market-ордер (тільки post_only limit)."""
    client = MockExchangeClient()
    adapter = _make_adapter(client)
    adapter._quote(pd.Timestamp("2025-01-01"), want=1, p1=100.0, p2=50.0)
    assert all(c["type"] == "limit" and c["post_only"] for c in client.created_calls)


def test_both_filled_books_position() -> None:
    """Обидві ноги заповнились → позиція заброньована, have встановлено."""
    client = MockExchangeClient()
    adapter = _make_adapter(client, legging="strict_both")
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    adapter._quote(ts0, want=1, p1=100.0, p2=50.0)
    lo1, lo2 = adapter._live_pending
    # симулюємо філ обох ніг на біржі
    client.fill_order(lo1.exchange_order_id, 100.0)
    client.fill_order(lo2.exchange_order_id, 50.0)
    # on_bar з сигналом hold → _resolve_pending опитує біржу → both filled
    action = adapter.on_bar(ts1, 101, 99, 100, 51, 49, 50, signal=1.0)
    assert "filled" in action
    assert adapter.have == 1
    assert len(adapter.account.positions) == 2
    assert adapter._live_pending is None


def test_neither_filled_timeout() -> None:
    """Жодна нога не заповнилась → після wait_bars скасувати, unfilled."""
    client = MockExchangeClient()
    adapter = _make_adapter(client, legging="strict_both")
    adapter.wait_bars = 1
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    adapter._quote(ts0, want=1, p1=100.0, p2=50.0)
    # не заповнюємо → on_bar опитує → neither → timeout (wait_bars=1), потім re-quote
    action = adapter.on_bar(ts1, 101, 99, 100, 51, 49, 50, signal=1.0)
    assert "unfilled:timeout" in action
    # обидва попередні скасовані; новий re-quote створив новий pending
    assert len(client.canceled_oids) == 2
    assert "quoted want=1" in action  # re-quote після timeout


def test_strict_both_partial_neither() -> None:
    """strict_both: одна нога заповнилась, інша ні → all-or-none (ні)."""
    client = MockExchangeClient()
    adapter = _make_adapter(client, legging="strict_both")
    adapter.wait_bars = 5
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    adapter._quote(ts0, want=1, p1=100.0, p2=50.0)
    lo1, lo2 = adapter._live_pending
    client.fill_order(lo1.exchange_order_id)  # лише перша
    action = adapter.on_bar(ts1, 101, 99, 100, 51, 49, 50, signal=1.0)
    # strict_both: partial → neither; чекає (wait_bars=5)
    assert "pending" in action
    assert adapter._live_pending is not None  # ще живе


def test_chase_mode_one_filled_other_chased() -> None:
    """chase: одна нога maker-філ, інша → taker chase (якщо drift ≤ max)."""
    client = MockExchangeClient()
    adapter = _make_adapter(client, legging="chase")
    ts0 = pd.Timestamp("2025-01-01 00:00")
    ts1 = pd.Timestamp("2025-01-01 01:00")
    adapter._quote(ts0, want=1, p1=100.0, p2=50.0)
    lo1, lo2 = adapter._live_pending
    client.fill_order(lo1.exchange_order_id, 100.0)  # leg1 maker-філ
    # mid2 ≈ limit2 (drift ≈ 0) → chase leg2
    action = adapter.on_bar(ts1, 101, 99, 100, 50.5, 49.5, 50.0, signal=1.0)
    assert "filled:chase" in action
    assert adapter.have == 1


def test_cancel_pending_cancels_real_orders() -> None:
    """cancel_pending (shutdown) скасує реальні ордери ніг."""
    client = MockExchangeClient()
    adapter = _make_adapter(client)
    adapter._quote(pd.Timestamp("2025-01-01"), want=1, p1=100.0, p2=50.0)
    assert len(client.orders) == 2
    adapter.cancel_pending(reason="shutdown")
    assert adapter._live_pending is None
    assert len(client.canceled_oids) == 2


def test_start_live_empty_ok() -> None:
    """Порожній локальний + порожня біржа → ok."""
    client = MockExchangeClient()
    adapter = _make_adapter(client)
    assert adapter.start_live() == "ok"
    assert adapter._reconciled is True


def test_start_live_hydrate_from_exchange() -> None:
    """Порожній локальний + є ноги на біржі → гідратація have."""
    client = MockExchangeClient()
    client.positions = [
        {"symbol": "AAA", "side": "short", "contracts": 10.0},
        {"symbol": "BBB", "side": "long", "contracts": 20.0},
    ]
    adapter = _make_adapter(client)
    adapter.start_live()
    assert adapter.have == 1  # leg1 short → want=+1
    assert len(adapter.account.positions) == 2


def test_start_live_drift_killswitch() -> None:
    """Локальний має позицію, біржа іншу → KillSwitch."""
    client = MockExchangeClient()
    client.positions = [{"symbol": "AAA", "side": "long", "contracts": 10.0}]
    adapter = _make_adapter(client)
    # спершу створимо локальну позицію (розбіжність)
    adapter.account.open_position("AAA/BBB:AAA", "short", 10.0, 100.0, pd.Timestamp("2025-01-01"), is_maker=True)
    with pytest.raises(KillSwitch, match="reconcile"):
        adapter.start_live()


def test_reduce_only_on_exit() -> None:
    """Вихід (want=0) → ордери з reduceOnly=True."""
    client = MockExchangeClient()
    adapter = _make_adapter(client)
    # спершу відкриємо позицію (заброньовано)
    adapter.account.open_position("AAA/BBB:AAA", "short", 10.0, 100.0, pd.Timestamp("2025-01-01"), is_maker=True)
    adapter.account.open_position("AAA/BBB:BBB", "long", 20.0, 50.0, pd.Timestamp("2025-01-01"), is_maker=True)
    adapter.have = 1
    client.created_calls.clear()
    # want=0 → вихід
    adapter._quote(pd.Timestamp("2025-01-01 01:00"), want=0, p1=100.0, p2=50.0)
    assert all(c["reduce_only"] for c in client.created_calls), "вихід має бути reduce_only"


def test_reconcile_runtime_no_drift() -> None:
    """reconcile_runtime: без drift → без KillSwitch."""
    client = MockExchangeClient()
    adapter = _make_adapter(client)
    adapter.start_live()
    adapter.reconcile_runtime()  # не кидає


def test_reconcile_runtime_drift_killswitch() -> None:
    """reconcile_runtime: drift → KillSwitch."""
    client = MockExchangeClient()
    adapter = _make_adapter(client)
    adapter.start_live()
    # локальна позиція є, біржа порожня → drift
    adapter.account.open_position("AAA/BBB:AAA", "short", 10.0, 100.0, pd.Timestamp("2025-01-01"), is_maker=True)
    client.positions = []  # біржа порожня
    with pytest.raises(KillSwitch, match="drift"):
        adapter.reconcile_runtime()


def test_taker_rejected_on_init() -> None:
    """is_maker=False → ValueError (taker заборонено на вході)."""
    client = MockExchangeClient()
    acc = PaperAccount(10_000.0, taker_fee=0.0005, maker_fee=0.0002)
    with pytest.raises(ValueError, match="лише maker"):
        PairsLiveAdapter("AAA", "BBB", PairsArb(lookback=20), acc, client=client, is_maker=False, coint_kill=False)


def test_idempotent_client_order_ids() -> None:
    """Кожен розміщений ордер має унікальний client_order_id."""
    client = MockExchangeClient()
    adapter = _make_adapter(client)
    adapter._quote(pd.Timestamp("2025-01-01"), want=1, p1=100.0, p2=50.0)
    lo1, lo2 = adapter._live_pending
    assert lo1.client_order_id != lo2.client_order_id
    assert lo1.client_order_id.startswith("shp-")


def test_pairs_live_runner_requires_live_credentials(monkeypatch) -> None:
    """PairsLiveRunner — fail-closed: без ключів не стартує."""
    from scalper_hft.config import Settings
    from scalper_hft.live.pairs_runner import PairsLiveRunner

    # без ключів → RuntimeError
    monkeypatch.setattr(
        "scalper_hft.live.pairs_runner.get_settings",
        lambda: Settings(
            binance_api_key="",
            binance_api_secret="",
            dry_run=False,
        ),
    )
    with pytest.raises(RuntimeError, match="BINANCE_API_KEY"):
        PairsLiveRunner("AAA", "BBB", client=MockExchangeClient(), restore=False)


def test_pairs_live_runner_rejects_dry_run(monkeypatch) -> None:
    """PairsLiveRunner — DRY_RUN=true заборонено (це paper-режим)."""
    from scalper_hft.config import Settings
    from scalper_hft.live.pairs_runner import PairsLiveRunner

    monkeypatch.setattr(
        "scalper_hft.live.pairs_runner.get_settings",
        lambda: Settings(
            binance_api_key="k",
            binance_api_secret="s",
            dry_run=True,
        ),
    )
    with pytest.raises(RuntimeError, match="DRY_RUN=true заборонено"):
        PairsLiveRunner("AAA", "BBB", client=MockExchangeClient(), restore=False)


def test_pairs_live_runner_requires_client(monkeypatch) -> None:
    """PairsLiveRunner без ExchangeClient → RuntimeError."""
    from scalper_hft.config import Settings
    from scalper_hft.live.pairs_runner import PairsLiveRunner

    monkeypatch.setattr(
        "scalper_hft.live.pairs_runner.get_settings",
        lambda: Settings(
            binance_api_key="k",
            binance_api_secret="s",
            dry_run=False,
        ),
    )
    with pytest.raises(RuntimeError, match="ExchangeClient"):
        PairsLiveRunner("AAA", "BBB", client=None, restore=False)
