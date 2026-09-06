"""Інтерактивний Telegram Bot (python-telegram-bot v20+, async).

Команди:
    /status           — баланс, PnL, відкриті позиції
    /trades [N]       — останні N угод (default 10, max 50)
    /bots             — список активних ботів і їх стан
    /pause [flag]     — no_new_entries або pause (потребує PIN)
    /resume           — зняти no_new_entries/pause (потребує PIN)
    /equity           — equity curve PNG (matplotlib)
    /risk             — поточне використання ризик-лімітів
    /regime           — режим ринку з останнього snapshot
    /help             — список команд

Безпека:
    - TELEGRAM_ALLOWED_CHAT_IDS: whitelist chat_id через кому.
      Якщо порожнє — дозволяється лише TELEGRAM_CHAT_ID.
    - /pause та /resume потребують: /pause <PIN>
    - Rate limit: ≤30 команд за хвилину на chat_id.

Запуск:
    uv run python -m scalper_hft.cli telegram-bot start
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from scalper_hft.config import get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── Константи ────────────────────────────────────────────────────────────────

RATE_LIMIT_MAX = 30   # максимум команд
RATE_LIMIT_SEC = 60   # за цей час (секунди)

_HELP_TEXT = """
🤖 *Scalper HFT Bot — команди*

📊 *Дані*
/status — баланс, позиції, PnL
/trades [N] — останні N угод (max 50)
/bots — статус усіх ботів
/equity — equity curve (PNG)
/risk — ризик-ліміти
/regime — режим ринку

⚙️ *Управління (потребує PIN)*
/pause <PIN> — зупинити нові входи
/resume <PIN> — відновити торгівлю

/help — цей список
""".strip()


# ─── Rate limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """Sliding-window rate limiter на chat_id."""

    def __init__(self, max_calls: int = RATE_LIMIT_MAX, window_sec: int = RATE_LIMIT_SEC):
        self._max = max_calls
        self._window = window_sec
        self._calls: dict[int, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, chat_id: int) -> bool:
        now = time.monotonic()
        with self._lock:
            dq = self._calls[chat_id]
            # Видалити застарілі записи
            while dq and now - dq[0] > self._window:
                dq.popleft()
            if len(dq) >= self._max:
                return False
            dq.append(now)
            return True


# ─── Bot ──────────────────────────────────────────────────────────────────────

class TelegramBotServer:
    """Async Telegram Bot для моніторингу та управління ботами.

    Параметри:
        store_path: шлях до SQLite (results/paper_pairs.sqlite)
        control_path: шлях до control.json
    """

    def __init__(
        self,
        store_path: Path | str | None = None,
        control_path: Path | str | None = None,
    ) -> None:
        from scalper_hft.live.store import PaperStore

        self._settings = get_settings()
        self._store = PaperStore(store_path)
        self._control_path = Path(control_path) if control_path else Path("results") / "control.json"
        self._limiter = _RateLimiter()
        self._app: object | None = None  # telegram.ext.Application

    # ─── Whitelist ────────────────────────────────────────────────────────────

    def _allowed_ids(self) -> set[int]:
        """Повернути whitelist chat_id з конфігурації."""
        ids: set[int] = set()
        # Додати основний TELEGRAM_CHAT_ID якщо є
        main_id = (self._settings.telegram_chat_id or "").strip()
        if main_id:
            try:
                ids.add(int(main_id))
            except ValueError:
                pass
        # Додати TELEGRAM_ALLOWED_CHAT_IDS
        extra = (self._settings.telegram_allowed_chat_ids or "").strip()
        for part in extra.split(","):
            part = part.strip()
            if part:
                try:
                    ids.add(int(part))
                except ValueError:
                    logger.warning("TELEGRAM_ALLOWED_CHAT_IDS: невалідний chat_id %r", part)
        return ids

    def _is_allowed(self, chat_id: int) -> bool:
        allowed = self._allowed_ids()
        if not allowed:
            logger.debug("Whitelist порожній — відхиляю всіх")
            return False
        return chat_id in allowed

    # ─── PIN перевірка ────────────────────────────────────────────────────────

    def _check_pin(self, provided: str) -> bool:
        pin = (self._settings.telegram_bot_pin or "").strip()
        if not pin:
            return False  # PIN не налаштовано → деструктивні команди вимкнені
        return provided.strip() == pin

    # ─── Форматери ────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_float(val: float | None, decimals: int = 2, suffix: str = "") -> str:
        if val is None:
            return "N/A"
        return f"{val:+.{decimals}f}{suffix}"

    @staticmethod
    def _fmt_ts(ts_str: str) -> str:
        try:
            dt = pd.Timestamp(ts_str)
            return dt.strftime("%m-%d %H:%M")
        except Exception:
            return ts_str[:16]

    # ─── Команди ──────────────────────────────────────────────────────────────

    async def _cmd_help(self, update: object, ctx: object) -> None:
        await self._reply(update, _HELP_TEXT, parse_mode="Markdown")

    async def _cmd_status(self, update: object, ctx: object) -> None:
        chat_id = self._get_chat_id(update)
        if not self._guard(chat_id, update):
            return

        try:
            equity_df = self._store.recent_equity(limit=1)
            trades_df = self._store.all_trades()
            positions_open = trades_df[trades_df["exit_price"].isna()] if not trades_df.empty else pd.DataFrame()

            if equity_df.empty:
                await self._reply(update, "📊 Даних ще немає. Запустіть бота.")
                return

            row = equity_df.iloc[0]
            equity = float(row.get("equity", 0))
            realized = float(row.get("realized_pnl", 0))
            n_pos = len(positions_open)

            lines = [
                "📊 *Статус торгівлі*",
                f"💰 Equity: `${equity:.2f}`",
                f"📈 Realized PnL: `{realized:+.2f} USDT`",
                f"📌 Відкритих позицій: `{n_pos}`",
                f"⏰ Оновлено: `{self._fmt_ts(str(row.get('ts', '')))}`",
            ]
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")

        except Exception as exc:
            logger.exception("cmd_status error")
            await self._reply(update, f"⚠️ Помилка: {exc}")

    async def _cmd_trades(self, update: object, ctx: object) -> None:
        chat_id = self._get_chat_id(update)
        if not self._guard(chat_id, update):
            return

        # Парсинг N з аргументів
        args = self._get_args(ctx)
        try:
            n = min(int(args[0]), 50) if args else 10
        except (ValueError, IndexError):
            n = 10

        try:
            df = self._store.all_trades()
            if df.empty:
                await self._reply(update, "📋 Угод ще немає.")
                return

            df = df.tail(n)
            lines = ["📋 *Останні угоди*\n"]
            for _, row in df.iterrows():
                pnl = row.get("pnl")
                pnl_str = f"{float(pnl):+.2f}" if pnl is not None and pd.notna(pnl) else "open"
                symbol = row.get("symbol", "?")
                side = row.get("side", "?")
                ts = self._fmt_ts(str(row.get("ts", "")))
                lines.append(f"`{ts}` {symbol} {side.upper()} PnL={pnl_str}")

            await self._reply(update, "\n".join(lines), parse_mode="Markdown")

        except Exception as exc:
            logger.exception("cmd_trades error")
            await self._reply(update, f"⚠️ Помилка: {exc}")

    async def _cmd_bots(self, update: object, ctx: object) -> None:
        chat_id = self._get_chat_id(update)
        if not self._guard(chat_id, update):
            return

        try:
            # Намагаємось прочитати таблицю bots (є в UnifiedTradeStore Sprint 2)
            try:
                import sqlite3
                conn = sqlite3.connect(self._store.path)
                df = pd.read_sql_query("SELECT bot_id, exchange, symbol, interval, strategy, mode, status, last_heartbeat FROM bots", conn)
                conn.close()
            except Exception:
                df = pd.DataFrame()

            if df.empty:
                # Показати snapshot якщо є
                snapshot = self._store.load_runtime()
                if snapshot:
                    status = "running" if snapshot else "unknown"
                    await self._reply(update, f"🤖 Paper runner: `{status}`\n_(таблиця bots з'явиться в Sprint 2)_", parse_mode="Markdown")
                else:
                    await self._reply(update, "🤖 Активних ботів не знайдено.")
                return

            lines = ["🤖 *Активні боти*\n"]
            for _, row in df.iterrows():
                status_emoji = "🟢" if row["status"] == "running" else "🔴" if row["status"] == "error" else "🟡"
                hb = self._fmt_ts(str(row.get("last_heartbeat", ""))) if row.get("last_heartbeat") else "N/A"
                lines.append(
                    f"{status_emoji} `{row['bot_id']}` | {row['exchange']} {row['symbol']} {row['interval']}\n"
                    f"   стратегія: {row['strategy']} | mode: {row['mode']} | heartbeat: {hb}"
                )
            await self._reply(update, "\n".join(lines), parse_mode="Markdown")

        except Exception as exc:
            logger.exception("cmd_bots error")
            await self._reply(update, f"⚠️ Помилка: {exc}")

    async def _cmd_pause(self, update: object, ctx: object) -> None:
        chat_id = self._get_chat_id(update)
        if not self._guard(chat_id, update):
            return

        args = self._get_args(ctx)
        pin = args[0] if args else ""
        if not self._check_pin(pin):
            pin_configured = bool((self._settings.telegram_bot_pin or "").strip())
            if not pin_configured:
                await self._reply(update, "⛔ TELEGRAM\\_BOT\\_PIN не налаштовано в .env", parse_mode="Markdown")
            else:
                await self._reply(update, "⛔ Невірний PIN. Використання: `/pause <PIN>`", parse_mode="Markdown")
            return

        try:
            self._write_control(no_new_entries=True)
            await self._reply(update, "⏸ *Пауза увімкнена* — нові входи заблоковано.\nВідкриті позиції залишаються.\n/resume <PIN> для відновлення.", parse_mode="Markdown")
        except Exception as exc:
            await self._reply(update, f"⚠️ Помилка запису control.json: {exc}")

    async def _cmd_resume(self, update: object, ctx: object) -> None:
        chat_id = self._get_chat_id(update)
        if not self._guard(chat_id, update):
            return

        args = self._get_args(ctx)
        pin = args[0] if args else ""
        if not self._check_pin(pin):
            pin_configured = bool((self._settings.telegram_bot_pin or "").strip())
            if not pin_configured:
                await self._reply(update, "⛔ TELEGRAM\\_BOT\\_PIN не налаштовано в .env", parse_mode="Markdown")
            else:
                await self._reply(update, "⛔ Невірний PIN. Використання: `/resume <PIN>`", parse_mode="Markdown")
            return

        try:
            self._write_control(no_new_entries=False, pause=False)
            await self._reply(update, "▶️ *Торгівля відновлена* — нові входи дозволені.", parse_mode="Markdown")
        except Exception as exc:
            await self._reply(update, f"⚠️ Помилка запису control.json: {exc}")

    async def _cmd_equity(self, update: object, ctx: object) -> None:
        chat_id = self._get_chat_id(update)
        if not self._guard(chat_id, update):
            return

        try:
            df = self._store.all_equity()
            if df.empty:
                await self._reply(update, "📈 Equity даних ще немає.")
                return

            # Малюємо PNG через matplotlib
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            df["ts"] = pd.to_datetime(df["ts"])
            df = df.sort_values("ts")

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")

            # Equity по парах якщо є
            if "pair" in df.columns:
                for pair, grp in df.groupby("pair"):
                    ax.plot(grp["ts"], grp["equity"], linewidth=1.5, label=pair)
            else:
                ax.plot(df["ts"], df["equity"], color="#00d4ff", linewidth=2, label="equity")

            # Стиль
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.set_title("📈 Equity Curve", color="white", fontsize=14)
            ax.set_ylabel("USDT", color="#aaa")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.grid(True, alpha=0.2, color="#444")
            if "pair" in df.columns and df["pair"].nunique() > 1:
                ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

            # Зберегти в буфер
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)

            # Відправити фото
            await self._send_photo(update, buf, caption="📈 Equity Curve")

        except Exception as exc:
            logger.exception("cmd_equity error")
            await self._reply(update, f"⚠️ Помилка побудови графіку: {exc}")

    async def _cmd_risk(self, update: object, ctx: object) -> None:
        chat_id = self._get_chat_id(update)
        if not self._guard(chat_id, update):
            return

        try:
            snapshot = self._store.load_runtime()
            ctrl = self._read_control()

            lines = ["⚠️ *Ризик-ліміти*\n"]

            # Control state
            if ctrl.get("pause"):
                lines.append("🔴 Стан: PAUSE")
            elif ctrl.get("no_new_entries"):
                lines.append("🟡 Стан: NO NEW ENTRIES")
            else:
                lines.append("🟢 Стан: Торгівля активна")

            # З snapshot
            if snapshot:
                account = snapshot.get("account", {})
                equity = account.get("equity", None)
                day_start = account.get("day_start_equity", None)
                if equity and day_start:
                    dd = (equity - day_start) / day_start * 100
                    lines.append(f"📉 Денний DD: `{dd:+.2f}%` (ліміт: `{self._settings.daily_loss_limit * 100:.1f}%`)")

                n_consec = account.get("consecutive_losses", 0)
                lines.append(f"❌ Послідовних збитків: `{n_consec}` (макс: `{self._settings.max_consecutive_losses}`)")

            await self._reply(update, "\n".join(lines), parse_mode="Markdown")

        except Exception as exc:
            logger.exception("cmd_risk error")
            await self._reply(update, f"⚠️ Помилка: {exc}")

    async def _cmd_regime(self, update: object, ctx: object) -> None:
        chat_id = self._get_chat_id(update)
        if not self._guard(chat_id, update):
            return

        try:
            snapshot = self._store.load_runtime()
            if not snapshot:
                await self._reply(update, "🔍 Snapshot недоступний. Запустіть бота з `--daemon`.")
                return

            regime_info = snapshot.get("regime", {})
            if not regime_info:
                await self._reply(update, "🔍 Дані про режим відсутні в snapshot.")
                return

            lines = ["🌐 *Режими ринку*\n"]
            if isinstance(regime_info, dict):
                for sym, info in regime_info.items():
                    structure = info.get("structure", "?")
                    vol = info.get("vol", "?")
                    emoji = {"trend_up": "📈", "trend_down": "📉", "range": "↔️"}.get(structure, "❓")
                    lines.append(f"{emoji} `{sym}`: {structure} | vol={vol}")
            else:
                lines.append(str(regime_info))

            await self._reply(update, "\n".join(lines), parse_mode="Markdown")

        except Exception as exc:
            logger.exception("cmd_regime error")
            await self._reply(update, f"⚠️ Помилка: {exc}")

    # ─── Допоміжні методи ────────────────────────────────────────────────────

    def _guard(self, chat_id: int | None, update: object) -> bool:
        """Перевірити whitelist і rate limit. Повертає True якщо дозволено."""
        if chat_id is None:
            return False
        if not self._is_allowed(chat_id):
            import asyncio
            asyncio.ensure_future(self._reply(update, "⛔ Доступ заборонено."))
            return False
        if not self._limiter.is_allowed(chat_id):
            import asyncio
            asyncio.ensure_future(self._reply(update, f"⏳ Rate limit: максимум {RATE_LIMIT_MAX} команд за {RATE_LIMIT_SEC}с."))
            return False
        return True

    @staticmethod
    def _get_chat_id(update: object) -> int | None:
        try:
            return update.effective_chat.id  # type: ignore[attr-defined]
        except AttributeError:
            return None

    @staticmethod
    def _get_args(ctx: object) -> list[str]:
        try:
            return list(ctx.args or [])  # type: ignore[attr-defined]
        except AttributeError:
            return []

    @staticmethod
    async def _reply(update: object, text: str, parse_mode: str | None = None) -> None:
        try:
            await update.message.reply_text(  # type: ignore[attr-defined]
                text[:4000],
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("Telegram reply error: %s", exc)

    @staticmethod
    async def _send_photo(update: object, buf: io.BytesIO, caption: str = "") -> None:
        try:
            await update.message.reply_photo(  # type: ignore[attr-defined]
                photo=buf,
                caption=caption[:1024],
            )
        except Exception as exc:
            logger.warning("Telegram send_photo error: %s", exc)

    def _write_control(self, **kwargs: bool) -> None:
        """Атомарно оновити control.json."""
        path = self._control_path
        path.parent.mkdir(parents=True, exist_ok=True)
        current: dict = {}
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
        current.update(kwargs)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        logger.info("control.json оновлено: %s", kwargs)

    def _read_control(self) -> dict:
        """Прочитати поточний control.json як dict."""
        if not self._control_path.exists():
            return {}
        try:
            return json.loads(self._control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ─── Запуск ──────────────────────────────────────────────────────────────

    def _build_app(self) -> object:
        """Зібрати telegram.ext.Application з CommandHandler'ами."""
        from telegram.ext import Application, CommandHandler  # type: ignore[import]

        token = (self._settings.telegram_bot_token or "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN не налаштовано в .env")

        app = (
            Application.builder()
            .token(token)
            .build()
        )

        handlers = [
            ("help", self._cmd_help),
            ("start", self._cmd_help),
            ("status", self._cmd_status),
            ("trades", self._cmd_trades),
            ("bots", self._cmd_bots),
            ("pause", self._cmd_pause),
            ("resume", self._cmd_resume),
            ("equity", self._cmd_equity),
            ("risk", self._cmd_risk),
            ("regime", self._cmd_regime),
        ]
        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, handler))

        return app

    def run_polling(self) -> None:
        """Запустити бота (блокуючий, для daemon thread або main)."""
        app = self._build_app()
        self._app = app
        allowed = self._allowed_ids()
        logger.info(
            "Telegram Bot стартує. Whitelist chat_id: %s",
            sorted(allowed) if allowed else "не налаштовано",
        )
        app.run_polling(drop_pending_updates=True)  # type: ignore[attr-defined]

    def stop(self) -> None:
        """Graceful shutdown бота."""
        if self._app is not None:
            try:
                self._app.stop()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("Bot stop error (ignored): %s", exc)


def start_bot_thread(
    store_path: Path | str | None = None,
    control_path: Path | str | None = None,
) -> tuple[TelegramBotServer, threading.Thread]:
    """Запустити бота в daemon-потоці. Повертає (server, thread)."""
    server = TelegramBotServer(store_path=store_path, control_path=control_path)
    t = threading.Thread(target=server.run_polling, daemon=True, name="telegram-bot")
    t.start()
    return server, t


__all__ = ["TelegramBotServer", "start_bot_thread"]
