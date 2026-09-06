"""Тести для TelegramBotServer (мок telegram API, без мережевих запитів).

Стратегія: мок telegram.Update + ContextTypes.DEFAULT_TYPE.
Всі asyncio.run() виконуються через pytest-asyncio або синхронно через asyncio.get_event_loop().
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest


# ─── Хелпери ──────────────────────────────────────────────────────────────────

def _make_update(chat_id: int = 123456789, text: str = "/status") -> MagicMock:
    """Створити мок telegram.Update з потрібним chat_id."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock(return_value=None)
    update.message.reply_photo = AsyncMock(return_value=None)
    update.message.text = text
    return update


def _make_ctx(args: list[str] | None = None) -> MagicMock:
    """Створити мок telegram.ext.ContextTypes."""
    ctx = MagicMock()
    ctx.args = args or []
    return ctx


def _make_server(
    tmp_path: Path,
    allowed_ids: str = "123456789",
    pin: str = "9999",
) -> object:
    """Створити TelegramBotServer з тимчасовим SQLite та конфігурацією."""
    from scalper_hft.live.telegram_bot import TelegramBotServer

    store_path = tmp_path / "test_store.sqlite"
    control_path = tmp_path / "control.json"

    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "123456789",
        "TELEGRAM_ALLOWED_CHAT_IDS": allowed_ids,
        "TELEGRAM_BOT_PIN": pin,
    }):
        # Скинути синглтон щоб Settings перечиталась
        import scalper_hft.config as cfg_module
        cfg_module._settings = None
        server = TelegramBotServer(
            store_path=store_path,
            control_path=control_path,
        )
        cfg_module._settings = None  # cleanup
        return server


def _run(coro: object) -> object:
    """Синхронно виконати корутину."""
    return asyncio.get_event_loop().run_until_complete(coro)  # type: ignore[arg-type]


# ─── Тести whitelist ───────────────────────────────────────────────────────────

def test_allowed_ids_from_env(tmp_path: Path) -> None:
    """TELEGRAM_ALLOWED_CHAT_IDS парсується правильно."""
    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "t",
        "TELEGRAM_CHAT_ID": "111",
        "TELEGRAM_ALLOWED_CHAT_IDS": "222, 333",
        "TELEGRAM_BOT_PIN": "",
    }):
        import scalper_hft.config as cfg_module
        cfg_module._settings = None
        from scalper_hft.live.telegram_bot import TelegramBotServer
        server = TelegramBotServer(tmp_path / "s.sqlite", tmp_path / "c.json")
        ids = server._allowed_ids()
        assert 111 in ids
        assert 222 in ids
        assert 333 in ids
        cfg_module._settings = None


def test_unknown_chat_id_rejected(tmp_path: Path) -> None:
    """Незнайомий chat_id → _is_allowed повертає False."""
    server = _make_server(tmp_path, allowed_ids="123456789")
    assert not server._is_allowed(999999999)
    assert server._is_allowed(123456789)


def test_empty_whitelist_rejects_all(tmp_path: Path) -> None:
    """Порожній whitelist → всі відхиляються."""
    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "t",
        "TELEGRAM_CHAT_ID": "",
        "TELEGRAM_ALLOWED_CHAT_IDS": "",
        "TELEGRAM_BOT_PIN": "",
    }):
        import scalper_hft.config as cfg_module
        cfg_module._settings = None
        from scalper_hft.live.telegram_bot import TelegramBotServer
        server = TelegramBotServer(tmp_path / "s.sqlite", tmp_path / "c.json")
        assert not server._is_allowed(123456789)
        cfg_module._settings = None


# ─── Тести PIN ─────────────────────────────────────────────────────────────────

def test_check_pin_correct(tmp_path: Path) -> None:
    server = _make_server(tmp_path, pin="1234")
    assert server._check_pin("1234") is True


def test_check_pin_wrong(tmp_path: Path) -> None:
    server = _make_server(tmp_path, pin="1234")
    assert server._check_pin("0000") is False


def test_check_pin_empty_disables(tmp_path: Path) -> None:
    """Якщо PIN не налаштовано — деструктивні команди вимкнені."""
    server = _make_server(tmp_path, pin="")
    assert server._check_pin("") is False
    assert server._check_pin("anything") is False


# ─── Тести rate limiter ────────────────────────────────────────────────────────

def test_rate_limiter_allows_under_limit() -> None:
    from scalper_hft.live.telegram_bot import _RateLimiter
    rl = _RateLimiter(max_calls=5, window_sec=60)
    for _ in range(5):
        assert rl.is_allowed(1)


def test_rate_limiter_blocks_over_limit() -> None:
    from scalper_hft.live.telegram_bot import _RateLimiter
    rl = _RateLimiter(max_calls=5, window_sec=60)
    for _ in range(5):
        rl.is_allowed(1)
    # 6-й запит заблокований
    assert not rl.is_allowed(1)


def test_rate_limiter_different_chats_independent() -> None:
    from scalper_hft.live.telegram_bot import _RateLimiter
    rl = _RateLimiter(max_calls=2, window_sec=60)
    rl.is_allowed(1)
    rl.is_allowed(1)
    assert not rl.is_allowed(1)   # chat 1 заблокований
    assert rl.is_allowed(2)       # chat 2 незалежний


# ─── Тести команд ─────────────────────────────────────────────────────────────

def test_cmd_status_no_data(tmp_path: Path) -> None:
    """/status без даних → повідомлення 'Даних ще немає'."""
    server = _make_server(tmp_path)
    update = _make_update(chat_id=123456789)
    ctx = _make_ctx()
    _run(server._cmd_status(update, ctx))
    update.message.reply_text.assert_called_once()
    text = update.message.reply_text.call_args[0][0]
    assert "немає" in text.lower() or "статус" in text.lower()


def test_cmd_status_with_data(tmp_path: Path) -> None:
    """/status з даними → equity відображається."""
    server = _make_server(tmp_path)
    # Записати тестові дані
    ts = pd.Timestamp.now(tz="UTC")
    server._store.log_equity(ts, "XRP/BTC", equity=1050.0, cash=950.0, realized_pnl=50.0)

    update = _make_update(chat_id=123456789)
    ctx = _make_ctx()
    _run(server._cmd_status(update, ctx))
    update.message.reply_text.assert_called_once()
    text = update.message.reply_text.call_args[0][0]
    assert "1050" in text or "Equity" in text


def test_cmd_trades_empty(tmp_path: Path) -> None:
    """/trades без угод → повідомлення 'угод ще немає'."""
    server = _make_server(tmp_path)
    update = _make_update(chat_id=123456789)
    ctx = _make_ctx()
    _run(server._cmd_trades(update, ctx))
    text = update.message.reply_text.call_args[0][0]
    assert "немає" in text.lower()


def test_cmd_trades_with_data(tmp_path: Path) -> None:
    """/trades 3 → не більше 3 рядків угод."""
    server = _make_server(tmp_path)
    ts = pd.Timestamp.now(tz="UTC")
    for i in range(5):
        server._store.log_trade(ts, "XRP/BTC", {
            "symbol": "XRPUSDT", "side": "buy",
            "size": 100.0, "entry_price": 0.5,
            "exit_price": 0.51 if i % 2 == 0 else None,
            "pnl": 1.0 if i % 2 == 0 else None,
            "type": "trade",
        })

    update = _make_update(chat_id=123456789)
    ctx = _make_ctx(args=["3"])
    _run(server._cmd_trades(update, ctx))
    text = update.message.reply_text.call_args[0][0]
    # Перевірити що є вміст
    assert "XRPUSDT" in text or "угоди" in text.lower()


def test_cmd_pause_wrong_pin(tmp_path: Path) -> None:
    """/pause з неправильним PIN → відмова."""
    server = _make_server(tmp_path, pin="1234")
    update = _make_update(chat_id=123456789)
    ctx = _make_ctx(args=["0000"])
    _run(server._cmd_pause(update, ctx))
    text = update.message.reply_text.call_args[0][0]
    assert "невірний" in text.lower() or "pin" in text.lower()


def test_cmd_pause_correct_pin_writes_control(tmp_path: Path) -> None:
    """/pause <PIN> → control.json оновлюється."""
    server = _make_server(tmp_path, pin="1234")
    update = _make_update(chat_id=123456789)
    ctx = _make_ctx(args=["1234"])
    _run(server._cmd_pause(update, ctx))

    control_path = tmp_path / "control.json"
    assert control_path.exists()
    data = json.loads(control_path.read_text())
    assert data.get("no_new_entries") is True


def test_cmd_resume_correct_pin_clears_control(tmp_path: Path) -> None:
    """/resume <PIN> → no_new_entries=False у control.json."""
    server = _make_server(tmp_path, pin="1234")
    # Спочатку встановити pause
    server._write_control(no_new_entries=True, pause=True)

    update = _make_update(chat_id=123456789)
    ctx = _make_ctx(args=["1234"])
    _run(server._cmd_resume(update, ctx))

    data = json.loads((tmp_path / "control.json").read_text())
    assert data.get("no_new_entries") is False
    assert data.get("pause") is False


def test_cmd_unknown_chat_blocked(tmp_path: Path) -> None:
    """Незнайомий chat_id → reply_text не викликається з корисним вмістом."""
    server = _make_server(tmp_path, allowed_ids="123456789")
    update = _make_update(chat_id=999999999)  # не в whitelist
    ctx = _make_ctx()
    _run(server._cmd_status(update, ctx))
    # reply_text може бути викликаний з "доступ заборонено"
    if update.message.reply_text.called:
        text = update.message.reply_text.call_args[0][0]
        assert "заборон" in text.lower() or "⛔" in text


def test_cmd_help_returns_text(tmp_path: Path) -> None:
    """/help → повертає список команд."""
    server = _make_server(tmp_path)
    update = _make_update(chat_id=123456789)
    ctx = _make_ctx()
    _run(server._cmd_help(update, ctx))
    text = update.message.reply_text.call_args[0][0]
    assert "/status" in text
    assert "/trades" in text
    assert "/pause" in text


# ─── Тести нових push-функцій telegram.py ────────────────────────────────────

def test_notify_regime_change_formats_text() -> None:
    """notify_regime_change формує текст без помилок."""
    from scalper_hft.live.telegram import notify_regime_change
    with patch("scalper_hft.live.telegram._creds", return_value=("tok", "123")):
        with patch("scalper_hft.live.telegram.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})
            notify_regime_change(
                "BTCUSDT", "range", "trend_up",
                weights={"supertrend": 0.45, "mean_reversion": 0.30}
            )
            assert mock_post.called
            payload = mock_post.call_args[1]["json"]["text"]
            assert "BTCUSDT" in payload
            assert "trend_up" in payload


def test_notify_heartbeat_formats_text() -> None:
    """notify_heartbeat формує текст без помилок."""
    from scalper_hft.live.telegram import notify_heartbeat
    with patch("scalper_hft.live.telegram._creds", return_value=("tok", "123")):
        with patch("scalper_hft.live.telegram.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})
            notify_heartbeat("binance", "paper", 1050.0, 5.0, 3, 24.5)
            assert mock_post.called
            payload = mock_post.call_args[1]["json"]["text"]
            assert "1050" in payload
            assert "paper" in payload.lower()


def test_notify_risk_block_formats_text() -> None:
    """notify_risk_block формує текст без помилок."""
    from scalper_hft.live.telegram import notify_risk_block
    with patch("scalper_hft.live.telegram._creds", return_value=("tok", "123")):
        with patch("scalper_hft.live.telegram.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})
            notify_risk_block("binance", "MaxDD перевищено", dd_pct=8.5)
            assert mock_post.called
            payload = mock_post.call_args[1]["json"]["text"]
            assert "MaxDD" in payload
            assert "8.5" in payload



# ─── Інтеграційний тест: control.json взаємодія ───────────────────────────────

def test_write_and_read_control(tmp_path: Path) -> None:
    """_write_control + _read_control — round-trip."""
    server = _make_server(tmp_path)
    server._write_control(no_new_entries=True)
    data = server._read_control()
    assert data["no_new_entries"] is True

    server._write_control(no_new_entries=False, pause=False)
    data = server._read_control()
    assert data["no_new_entries"] is False


def test_control_merges_existing_keys(tmp_path: Path) -> None:
    """_write_control доповнює існуючі ключі, не перезаписує всі."""
    server = _make_server(tmp_path)
    server._write_control(no_new_entries=True)
    server._write_control(pause=True)  # не повинен затерти no_new_entries
    data = server._read_control()
    assert data.get("no_new_entries") is True
    assert data.get("pause") is True
