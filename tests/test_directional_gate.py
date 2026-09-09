"""Тести directional overfitting-гейту у paper/paper-run (fail-closed).

Перевіряє, що cmd_paper / cmd_paper_run під REQUIRE_AUDIT_PASS=true:
- стартують, якщо є свіжий PASS-вердикт;
- fail-closed (raise RuntimeError), якщо вердикта немає / FAIL / протух.
За замовчуванням (REQUIRE_AUDIT_PASS=false) гейт не діє.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _settings(**kw):
    base = dict(
        dry_run=True,
        require_audit_pass=False,
        audit_max_age_days=30,
        taker_fee=0.0005,
        maker_fee=0.0002,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _args_paper():
    return SimpleNamespace(symbol="BTCUSDT", interval="1h", days=1, strategy="mean_reversion", param_dict={})


def _args_paper_run():
    return SimpleNamespace(
        symbol="BTCUSDT", interval="1h", strategy="mean_reversion", param_dict={}, iterations=1, sleep=0, notify=False
    )


def test_cmd_paper_gate_off_by_default(monkeypatch) -> None:
    """REQUIRE_AUDIT_PASS=false → гейт не діє (навіть без вердикта не падає)."""
    import scalper_hft.config as cfg
    from scalper_hft.cli import cmd_paper

    monkeypatch.setattr(cfg, "get_settings", lambda: _settings(require_audit_pass=False))
    # не має падати на гейті; далі піде завантаження даних — патчимо його
    monkeypatch.setattr("scalper_hft.cli._load_klines", lambda *a, **k: __import__("pandas").DataFrame())
    raised = False
    try:
        cmd_paper(_args_paper())
    except Exception as exc:  # noqa: BLE001
        # будь-яка помилка далі по коду допустима, але НЕ від гейту
        raised = "Overfitting-гейт" in str(exc)
    assert not raised


def test_cmd_paper_gate_blocks_without_verdict(monkeypatch) -> None:
    import scalper_hft.config as cfg
    from scalper_hft.cli import cmd_paper

    monkeypatch.setattr(cfg, "get_settings", lambda: _settings(require_audit_pass=True))
    monkeypatch.setattr("scalper_hft.live.audit_gate.latest_verdict", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Overfitting-гейт"):
        cmd_paper(_args_paper())


def test_cmd_paper_run_gate_blocks_without_verdict(monkeypatch) -> None:
    import scalper_hft.config as cfg
    from scalper_hft.cli import cmd_paper_run

    monkeypatch.setattr(cfg, "get_settings", lambda: _settings(require_audit_pass=True))
    monkeypatch.setattr("scalper_hft.live.audit_gate.latest_verdict", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Overfitting-гейт"):
        cmd_paper_run(_args_paper_run())


def test_cmd_paper_gate_passes_with_fresh_verdict(monkeypatch) -> None:
    import scalper_hft.config as cfg
    from scalper_hft.cli import cmd_paper

    monkeypatch.setattr(cfg, "get_settings", lambda: _settings(require_audit_pass=True))
    monkeypatch.setattr(
        "scalper_hft.live.audit_gate.latest_verdict",
        lambda *a, **k: {"label": "PASS", "ts": "2026-09-01T00:00:00+00:00", "reasons": ""},
    )
    # далі піде завантаження даних — патчимо, щоб не падати на відсутності даних
    monkeypatch.setattr("scalper_hft.cli._load_klines", lambda *a, **k: __import__("pandas").DataFrame())
    raised = False
    try:
        cmd_paper(_args_paper())
    except RuntimeError as exc:
        raised = "Overfitting-гейт" in str(exc)
    except Exception:
        pass
    assert not raised
