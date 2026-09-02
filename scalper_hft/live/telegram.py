"""Telegram-сповіщення (Bot API).

Конфігурація: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID у .env
(ключі з trade-bots/.env; валідність перевірена через getMe).
Якщо ключів немає — функції тихо повертають False (no-op).
"""

from __future__ import annotations

import logging

import requests

from scalper_hft.config import get_settings

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


def _creds() -> tuple[str, str] | None:
    # .env завантажується через scalper_hft.config (load_dotenv у config.py);
    # раніше os.getenv читав процесні змінні, а .env — лише якщо config вже
    # імпортували → standalone-виклик мовчки втрачав алерти.
    s = get_settings()
    token = (s.telegram_bot_token or "").strip()
    chat_id = (s.telegram_chat_id or "").strip()
    if not token or not chat_id:
        return None
    return token, chat_id


def send_telegram(text: str, timeout: int = 10) -> bool:
    """Надіслати повідомлення. Повертає True при успіху; False — немає ключів/помилка."""
    creds = _creds()
    if creds is None:
        logger.debug("Telegram не налаштовано — пропускаю")
        return False
    token, chat_id = creds
    try:
        r = requests.post(
            _API.format(token=token),
            json={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True},
            timeout=timeout,
        )
        ok = r.status_code == 200 and r.json().get("ok", False)
        if not ok:
            logger.warning("Telegram send failed: %s %s", r.status_code, r.text[:200])
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram send error: %s", exc)
        return False


def notify_trade(symbol: str, action: str, price: float | None = None, pnl: float | None = None) -> bool:
    """Форматоване сповіщення про угоду."""
    parts = [f"🤖 *{symbol}*: {action}"]
    if price is not None:
        parts.append(f"ціна: {price:.4f}")
    if pnl is not None:
        parts.append(f"PnL: {pnl:+.2f} USDT")
    return send_telegram("\n".join(parts))


def notify_error(symbol: str, error: str) -> bool:
    return send_telegram(f"⚠️ *{symbol}* помилка: {error[:500]}")
