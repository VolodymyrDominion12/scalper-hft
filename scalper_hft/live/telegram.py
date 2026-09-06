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


def notify_regime_change(
    symbol: str,
    old_regime: str,
    new_regime: str,
    weights: dict[str, float] | None = None,
) -> bool:
    """💛 REGIME_CHANGE: symbol → new_regime з вагами стратегій."""
    emoji = {"trend_up": "📈", "trend_down": "📉", "range": "↔️"}.get(new_regime, "❓")
    text = f"💛 *REGIME* {symbol}: {old_regime} → {emoji} {new_regime}"
    if weights:
        w_str = " | ".join(f"{k}={v:.2f}" for k, v in weights.items())
        text += f"\nВаги: {w_str}"
    return send_telegram(text)


def notify_heartbeat(
    exchange: str,
    mode: str,
    equity: float,
    equity_pct: float,
    n_positions: int,
    uptime_hours: float,
) -> bool:
    """❤️ HEARTBEAT: бот живий + стан рахунку."""
    sign = "+" if equity_pct >= 0 else ""
    text = (
        f"❤️ *HEARTBEAT* | {mode.upper()} | {exchange}\n"
        f"Equity: `${equity:.2f}` ({sign}{equity_pct:.2f}%)\n"
        f"Позицій: `{n_positions}` | Uptime: `{uptime_hours:.1f}h`"
    )
    return send_telegram(text)


def notify_risk_block(
    exchange: str,
    reason: str,
    dd_pct: float | None = None,
) -> bool:
    """⚠️ RISK: ризик-ліміт спрацював, нові входи заблоковано."""
    text = f"⚠️ *RISK BLOCK* | {exchange}\n{reason}"
    if dd_pct is not None:
        text += f"\nDrawdown: `{dd_pct:.2f}%`"
    text += "\n/resume <PIN> для відновлення"
    return send_telegram(text)

