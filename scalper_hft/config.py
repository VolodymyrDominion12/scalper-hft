"""Конфігурація системи: змінні середовища з .env + розумні дефолти."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from scalper_hft.symbols import CANONICAL_SYMBOLS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Єдина точка конфігурації. Поля незмінні після створення."""

    # Binance
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    exchange: str = field(default_factory=lambda: os.getenv("EXCHANGE", "binance-testnet"))
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", True))

    # Transaction cost model (книга, гл. 5: комісії + slippage + market impact)
    maker_fee: float = field(default_factory=lambda: _env_float("MAKER_FEE", 0.0002))
    taker_fee: float = field(default_factory=lambda: _env_float("TAKER_FEE", 0.0005))
    slippage_bps: float = field(default_factory=lambda: _env_float("SLIPPAGE_BPS", 2.0))
    maker_execution: bool = field(default_factory=lambda: _env_bool("MAKER_EXECUTION", True))

    # Risk model
    position_pct: float = field(default_factory=lambda: _env_float("POSITION_PCT", 0.01))
    max_open_positions: int = field(default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 1))
    daily_loss_limit: float = field(default_factory=lambda: _env_float("DAILY_LOSS_LIMIT", 0.03))
    weekly_loss_limit: float = field(default_factory=lambda: _env_float("WEEKLY_LOSS_LIMIT", 0.06))
    max_consecutive_losses: int = field(default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 3))
    cooldown_losses: int = field(default_factory=lambda: _env_int("COOLDOWN_LOSSES", 2))
    cooldown_hours: float = field(default_factory=lambda: _env_float("COOLDOWN_HOURS", 12.0))
    cooldown_size_mult: float = field(default_factory=lambda: _env_float("COOLDOWN_SIZE_MULT", 0.5))
    corr_notional_cap: float = field(default_factory=lambda: _env_float("CORR_NOTIONAL_CAP", 0.40))
    pair_notional_pct: float = field(default_factory=lambda: _env_float("PAIR_NOTIONAL_PCT", 0.30))
    portfolio_notional_pct: float = field(default_factory=lambda: _env_float("PORTFOLIO_NOTIONAL_PCT", 0.60))
    max_losing_months: int = field(default_factory=lambda: _env_int("MAX_LOSING_MONTHS", 2))
    maker_fill_wait_bars: int = field(default_factory=lambda: _env_int("MAKER_FILL_WAIT_BARS", 1))

    # Дані
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    # Канонічний універсум за замовчуванням (scalper_hft/symbols.py); у .env —
    # DEFAULT_SYMBOLS може перевизначити його.
    default_symbols: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s.strip() for s in os.getenv("DEFAULT_SYMBOLS", ",".join(CANONICAL_SYMBOLS)).split(",") if s.strip()
        )
    )
    default_interval: str = field(default_factory=lambda: os.getenv("DEFAULT_INTERVAL", "1m"))
    download_retries: int = field(default_factory=lambda: _env_int("DOWNLOAD_RETRIES", 8))
    download_batch_delay: float = field(default_factory=lambda: _env_float("DOWNLOAD_BATCH_DELAY", 0.15))
    download_checkpoint_batches: int = field(default_factory=lambda: _env_int("DOWNLOAD_CHECKPOINT_BATCHES", 50))

    # Бекенд кешу даних: "parquet" (за замовчуванням, файли у data/) або
    # "postgres" (PostgreSQL у Docker — зручно для багатьох символів/таймфреймів).
    data_backend: str = field(default_factory=lambda: os.getenv("DATA_BACKEND", "parquet").strip().lower())
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: _env_int("POSTGRES_PORT", 5436))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "scalper"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "scalper"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "scalper"))
    postgres_dsn: str = field(default_factory=lambda: os.getenv("POSTGRES_DSN", ""))

    # Telegram (опційно; без ключів send — no-op)
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    # Telegram Bot — whitelist chat_id через кому; порожнє → лише telegram_chat_id
    telegram_allowed_chat_ids: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
    )
    # PIN для деструктивних команд (pause/stop/resume); порожнє → команди вимкнені
    telegram_bot_pin: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_PIN", "")
    )

    @property
    def postgres_conninfo(self) -> str:
        """libpq conninfo для psycopg: пріоритет має явний POSTGRES_DSN."""
        if self.postgres_dsn:
            return self.postgres_dsn
        return (
            f"host={self.postgres_host} port={self.postgres_port} dbname={self.postgres_db} "
            f"user={self.postgres_user} password={self.postgres_password}"
        )

    @property
    def slippage_frac(self) -> float:
        """Slippage як частка ціни (bps / 10_000)."""
        return self.slippage_bps / 10_000.0

    def fee(self, is_maker: bool) -> float:
        return self.maker_fee if is_maker else self.taker_fee

    @property
    def data_dir_abs(self) -> Path:
        p = self.data_dir
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p


_settings: Settings | None = None


def require_live_credentials(settings: Settings) -> None:
    """Fail-closed: live без ключів не стартує. Paper (dry_run) — без перевірки.

    У повідомленні немає значень ключів.
    """
    if settings.dry_run:
        return
    key = (settings.binance_api_key or "").strip()
    secret = (settings.binance_api_secret or "").strip()
    if not key or not secret:
        raise RuntimeError("Live режим (DRY_RUN=false) потребує BINANCE_API_KEY і BINANCE_API_SECRET")


def get_settings() -> Settings:
    """Кешований синглтон Settings (створюється один раз на процес)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
