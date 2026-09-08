"""Тести risk-моніторів Фази 3.4: DD-breaker, leverage cap, IntentStore."""

from __future__ import annotations

import pandas as pd
from scalper_hft.live.account import PaperAccount
from scalper_hft.live.intent_store import IntentStore
from scalper_hft.live.risk_gate import DrawdownBreaker, leverage_ok


class TestDrawdownBreaker:
    def test_triggers_below_peak(self) -> None:
        b = DrawdownBreaker(max_dd_pct=0.10)
        assert not b.check(10_000.0)
        assert not b.check(9_100.0)  # -9% від піку — ще ок
        assert b.check(8_900.0)  # -11% → halt

    def test_sticky_after_rebound(self) -> None:
        """Breaker липкий: відскок equity НЕ знімає halt (лише reset)."""
        b = DrawdownBreaker(max_dd_pct=0.10)
        b.check(10_000.0)
        assert b.check(8_900.0)
        assert b.check(9_900.0)  # відскок — все одно halt
        b.reset(9_900.0)
        assert not b.triggered
        assert not b.check(9_500.0)

    def test_peak_moves_up(self) -> None:
        b = DrawdownBreaker(max_dd_pct=0.10)
        b.check(10_000.0)
        b.check(12_000.0)  # новий пік
        assert not b.check(11_000.0)  # -8.3% від нового піку — ок
        assert b.check(10_700.0)  # -10.8% → halt


class TestLeverageCap:
    def test_within_cap(self) -> None:
        assert leverage_ok(10_000.0, 5_000.0, 10_000.0, max_leverage=3.0)

    def test_over_cap(self) -> None:
        assert not leverage_ok(25_000.0, 10_000.0, 10_000.0, max_leverage=3.0)

    def test_zero_equity_blocked(self) -> None:
        assert not leverage_ok(0.0, 100.0, 0.0, max_leverage=3.0)


def _engine(max_dd: float = 0.10):
    """PairsEngine з мінімальним оточенням (без store/біржі)."""
    from scalper_hft.live.pairs_engine import PairsEngine
    from scalper_hft.strategies.pairs_arb import PairsArb

    acc = PaperAccount(10_000.0, taker_fee=0.0005, maker_fee=0.0002)
    eng = PairsEngine(
        "AAA",
        "BBB",
        PairsArb(lookback=20),
        acc,
        wait_bars=1,
        is_maker=True,
        coint_kill=False,
    )
    eng.dd_breaker = DrawdownBreaker(max_dd_pct=max_dd, high_water=acc.equity)
    return eng


class TestEngineDDBreaker:
    def test_blocks_new_entries_when_triggered(self) -> None:
        eng = _engine()
        eng.dd_breaker.triggered = True
        ok, reason = eng._can_open(pd.Timestamp("2025-01-01"))
        assert not ok and "DD-breaker" in reason

    def test_auto_flatten_on_breach(self) -> None:
        """Просідання від піку > ліміту → on_bar закриває відкриті ноги."""
        eng = _engine(max_dd=0.05)
        ts0 = pd.Timestamp("2025-01-01 00:00")
        # відкриваємо пару локально (імітація заповнених ніг)
        eng.account.open_position("AAA/BBB:AAA", "short", 10.0, 100.0, ts0, is_maker=True)
        eng.account.open_position("AAA/BBB:BBB", "long", 20.0, 50.0, ts0, is_maker=True)
        eng.have = 1
        # пік уже 10_000; роняємо equity нижче 9_500 (обидві ноги в мінус)
        eng.account.mark({"AAA/BBB:AAA": 130.0, "AAA/BBB:BBB": 20.0})
        action = eng.on_bar(pd.Timestamp("2025-01-01 01:00"), 131, 129, 130, 21, 19, 20, signal=1.0)
        assert "dd_breaker" in action
        # pending на закриття виставлено (flatten-котирування reduce_only)
        assert eng.pending is not None
        assert all(o.reduce_only for o in eng.pending)


class TestEngineLeverageCap:
    def test_blocks_entry_over_leverage(self) -> None:
        eng = _engine()
        # після corr-капа size_pct стане ≤ 0.40 → нова пара = 2×0.40×equity;
        # кап плеча 0.5 → 0.8×equity ноціоналу не проходить
        eng.max_leverage = 0.5
        action = eng._quote(pd.Timestamp("2025-01-01"), want=1, p1=100.0, p2=50.0)
        assert "leverage cap" in action


class TestIntentStore:
    def test_put_get_pop_persisted(self, tmp_path) -> None:
        p = tmp_path / "intents.json"
        store = IntentStore(p)
        store.put("BTCUSDT:buy:entry:0", "sh-abc")
        # перечитуємо з диска — переживає "рестарт процесу"
        store2 = IntentStore(p)
        assert store2.get("BTCUSDT:buy:entry:0") == "sh-abc"
        store2.pop("BTCUSDT:buy:entry:0")
        assert IntentStore(p).get("BTCUSDT:buy:entry:0") is None

    def test_corrupt_file_starts_empty(self, tmp_path) -> None:
        p = tmp_path / "intents.json"
        p.write_text("{broken", encoding="utf-8")
        store = IntentStore(p)
        assert len(store) == 0

    def test_missing_file_ok(self, tmp_path) -> None:
        assert len(IntentStore(tmp_path / "nope.json")) == 0
