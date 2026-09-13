"""Тести податкового аудиту для України (Закон 10225-д, дослідження §8).

Дослідження §8: податок лише на прибуток; крипто-обмін не оподатковується;
оподатковується лише вихід у фіат (23%). Перевіряємо:
  - FIFO cost-basis
  - crypto→crypto обмін не taxable (cost-basis перенос)
  - crypto→fiat вихід taxable (23% на прибуток)
  - збиток не оподатковується
  - експорт CSV/JSON
  - CLI cmd_tax_report (через store mock)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scalper_hft.live.tax_audit import (
    FifoCostBasis,
    Fill,
    build_tax_report,
    export_csv,
    export_json,
)

# ── FifoCostBasis ────────────────────────────────────────────────────────────


class TestFifoCostBasis:
    def test_buy_then_sell_realizes_pnl(self) -> None:
        fb = FifoCostBasis()
        fb.buy("BTC", 1.0, 50_000.0)
        cost, proceeds, pnl = fb.sell("BTC", 1.0, 60_000.0)
        assert cost == pytest.approx(50_000.0)
        assert proceeds == pytest.approx(60_000.0)
        assert pnl == pytest.approx(10_000.0)

    def test_fifo_order(self) -> None:
        """FIFO: перший лот продається першим."""
        fb = FifoCostBasis()
        fb.buy("BTC", 1.0, 50_000.0)
        fb.buy("BTC", 1.0, 60_000.0)
        cost, _, pnl = fb.sell("BTC", 1.5, 70_000.0)
        # 1.0 × 50000 + 0.5 × 60000 = 80000
        assert cost == pytest.approx(80_000.0)
        assert pnl == pytest.approx(70_000 * 1.5 - 80_000.0)

    def test_sell_without_buy_zero_cost(self) -> None:
        fb = FifoCostBasis()
        cost, proceeds, pnl = fb.sell("BTC", 1.0, 60_000.0)
        assert cost == 0.0
        assert proceeds == pytest.approx(60_000.0)
        assert pnl == pytest.approx(60_000.0)

    def test_remaining_basis(self) -> None:
        fb = FifoCostBasis()
        fb.buy("BTC", 2.0, 50_000.0)
        fb.sell("BTC", 1.0, 60_000.0)
        assert fb.remaining_basis("BTC") == pytest.approx(50_000.0)

    def test_zero_qty_noop(self) -> None:
        fb = FifoCostBasis()
        fb.buy("BTC", 0.0, 50_000.0)
        assert fb.remaining_basis("BTC") == 0.0


# ── build_tax_report ────────────────────────────────────────────────────────


class TestBuildTaxReport:
    def test_crypto_to_crypto_not_taxable(self) -> None:
        """BTC→ETH обмін не оподатковується (лише cost-basis перенос)."""
        fills = [
            Fill("2026-01-01", "BTC", "buy", 1.0, 50_000.0, "USDT", "BTCUSDT"),
            # продаємо BTC за ETH (crypto→crypto, quote=ETH)
            Fill("2026-01-02", "BTC", "sell", 1.0, 20.0, "ETH", "BTCETH"),
        ]
        report = build_tax_report(fills)
        assert report.exchange_events_count == 1
        assert report.taxable_events_count == 0
        assert report.total_tax_estimate == 0.0

    def test_crypto_to_fiat_taxable(self) -> None:
        """BTC→USD (вихід у фіат) оподатковується 23% на прибуток."""
        fills = [
            Fill("2026-01-01", "BTC", "buy", 1.0, 50_000.0, "USDT", "BTCUSDT"),
            Fill("2026-01-02", "BTC", "sell", 1.0, 60_000.0, "USD", "BTCUSD"),
        ]
        report = build_tax_report(fills)
        assert report.taxable_events_count == 1
        assert report.exchange_events_count == 0
        assert report.total_realized_pnl == pytest.approx(10_000.0)
        assert report.total_tax_estimate == pytest.approx(10_000.0 * 0.23)

    def test_loss_not_taxed(self) -> None:
        """Збиток не оподатковується (tax=0 навіть при taxable event)."""
        fills = [
            Fill("2026-01-01", "BTC", "buy", 1.0, 60_000.0, "USDT", "BTCUSDT"),
            Fill("2026-01-02", "BTC", "sell", 1.0, 50_000.0, "USD", "BTCUSD"),
        ]
        report = build_tax_report(fills)
        assert report.taxable_events_count == 1
        assert report.total_realized_pnl == pytest.approx(-10_000.0)
        assert report.total_tax_estimate == 0.0  # збиток не оподатковується

    def test_usdt_quote_treated_as_crypto(self) -> None:
        """USDT — не фіат (crypto→crypto), отже не taxable."""
        fills = [
            Fill("2026-01-01", "BTC", "buy", 1.0, 50_000.0, "USDT", "BTCUSDT"),
            Fill("2026-01-02", "BTC", "sell", 1.0, 60_000.0, "USDT", "BTCUSDT"),
        ]
        report = build_tax_report(fills)
        assert report.taxable_events_count == 0  # USDT = crypto
        assert report.exchange_events_count == 1

    def test_uah_quote_is_fiat(self) -> None:
        fills = [
            Fill("2026-01-01", "BTC", "buy", 1.0, 50_000.0, "USDT", "BTCUSDT"),
            Fill("2026-01-02", "BTC", "sell", 1.0, 2_000_000.0, "UAH", "BTCUAH"),
        ]
        report = build_tax_report(fills)
        assert report.taxable_events_count == 1

    def test_multiple_buys_fifo(self) -> None:
        fills = [
            Fill("2026-01-01", "BTC", "buy", 0.5, 50_000.0, "USDT"),
            Fill("2026-01-02", "BTC", "buy", 0.5, 60_000.0, "USDT"),
            Fill("2026-01-03", "BTC", "sell", 0.5, 70_000.0, "USD"),
        ]
        report = build_tax_report(fills)
        # FIFO: 0.5 × 50000 = 25000 cost; 0.5 × 70000 = 35000 proceeds
        assert report.total_cost_basis == pytest.approx(25_000.0)
        assert report.total_proceeds == pytest.approx(35_000.0)
        assert report.total_realized_pnl == pytest.approx(10_000.0)

    def test_custom_fiat_set(self) -> None:
        """Користувач може розширити список фіатних валют."""
        fills = [
            Fill("2026-01-01", "BTC", "buy", 1.0, 50_000.0, "USDT"),
            Fill("2026-01-02", "BTC", "sell", 1.0, 60_000.0, "USDT", "BTCUSDT"),
        ]
        # якщо USDT — фіат, то taxable
        report = build_tax_report(fills, fiat_currencies=frozenset({"USDT"}))
        assert report.taxable_events_count == 1


# ── Експорт CSV/JSON ───────────────────────────────────────────────────────


class TestExport:
    def test_csv_export(self, tmp_path: Path) -> None:
        fills = [
            Fill("2026-01-01", "BTC", "buy", 1.0, 50_000.0, "USDT", "BTCUSDT"),
            Fill("2026-01-02", "BTC", "sell", 1.0, 60_000.0, "USD", "BTCUSD"),
        ]
        report = build_tax_report(fills)
        path = export_csv(report, tmp_path / "tax.csv")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "timestamp" in content
        assert "realized_pnl" in content
        assert "taxable" in content

    def test_json_export(self, tmp_path: Path) -> None:
        fills = [
            Fill("2026-01-01", "BTC", "buy", 1.0, 50_000.0, "USDT"),
            Fill("2026-01-02", "BTC", "sell", 1.0, 60_000.0, "USD"),
        ]
        report = build_tax_report(fills)
        path = export_json(report, tmp_path / "tax.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "summary" in data
        assert data["summary"]["taxable_events_count"] == 1
        assert len(data["events"]) == 1  # лише sell-подія

    def test_empty_report_export(self, tmp_path: Path) -> None:
        report = build_tax_report([])
        path = export_csv(report, tmp_path / "empty.csv")
        assert path.exists()
        # лише заголовок
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1


# ── CLI cmd_tax_report (через store mock) ──────────────────────────────────


class TestCliTaxReport:
    def test_cmd_tax_report_from_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """cmd_tax_report читає orders з PaperStore, будує звіт."""
        import argparse

        from scalper_hft.cli.ops import cmd_tax_report

        db = tmp_path / "test.sqlite"
        # створюємо store з тестовими ордерами
        from scalper_hft.live.store import PaperStore

        with PaperStore(db) as store:
            store.log_order(
                "2026-01-01", "BTCUSDT", "BTCUSDT", "buy", 1.0, 50000.0, "filled", "signal", 50000.0, 50000.0
            )
            store.log_order(
                "2026-01-02", "BTCUSDT", "BTCUSDT", "sell", 1.0, 60000.0, "filled", "exit", 60000.0, 60000.0
            )

        args = argparse.Namespace(year=2026, db=str(db), out=str(tmp_path / "tax_out"))
        cmd_tax_report(args)  # не повинен падати
        # перевіряємо що CSV створився
        csv_files = list((tmp_path / "tax_out").glob("tax_report_*.csv"))
        assert len(csv_files) == 1

    def test_cmd_tax_report_no_orders(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        import argparse

        from scalper_hft.cli.ops import cmd_tax_report
        from scalper_hft.live.store import PaperStore

        db = tmp_path / "empty.sqlite"
        with PaperStore(db):
            pass  # створити схему

        args = argparse.Namespace(year=0, db=str(db), out=str(tmp_path / "out"))
        cmd_tax_report(args)
        captured = capsys.readouterr()
        assert "Немає ордерів" in captured.out
