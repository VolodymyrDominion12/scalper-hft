"""Конфігурація системи: змінні середовища з .env + розумні дефолти."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from scalper_hft.symbols import CANONICAL_SYMBOLS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

#: Дефолтного JWT-секрета НЕМАЄ (раніше був захардкоджений dev-ключ у сорцях —
#: відомий ключ = відкритий доступ до керування ботом з публічного інтерфейсу).
#: Константа лишена для сумісності тестів/імпортам: її значення — порожній
#: рядок, який require_safe_api_bind відхиляє на non-localhost бінді.
DEFAULT_API_SECRET_KEY = ""

_LOCAL_API_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _parse_csv_tuple(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw.strip():
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _default_api_cors_origins() -> tuple[str, ...]:
    return (
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    )


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
    # Peak-to-trough DD breaker: просідання від історичного піку equity →
    # halt + авто-flatten (на відміну від daily/weekly — якір не зсувається)
    max_drawdown_pct: float = field(default_factory=lambda: _env_float("MAX_DRAWDOWN_PCT", 0.10))
    # Жорсткий ліміт плеча: сумарний ноціонал позицій ≤ equity × MAX_LEVERAGE
    max_leverage: float = field(default_factory=lambda: _env_float("MAX_LEVERAGE", 3.0))
    max_consecutive_losses: int = field(default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 3))
    cooldown_losses: int = field(default_factory=lambda: _env_int("COOLDOWN_LOSSES", 2))
    cooldown_hours: float = field(default_factory=lambda: _env_float("COOLDOWN_HOURS", 12.0))
    cooldown_size_mult: float = field(default_factory=lambda: _env_float("COOLDOWN_SIZE_MULT", 0.5))
    corr_notional_cap: float = field(default_factory=lambda: _env_float("CORR_NOTIONAL_CAP", 0.40))
    pair_notional_pct: float = field(default_factory=lambda: _env_float("PAIR_NOTIONAL_PCT", 0.30))
    portfolio_notional_pct: float = field(default_factory=lambda: _env_float("PORTFOLIO_NOTIONAL_PCT", 0.60))
    # Per-symbol notional cap + margin/liquidation proximity (1D): жорсткий ліміт
    # ноціоналу на одну монету (частка equity) + блок входів, коли сумарне плече
    # наближається до max_leverage (за `liquidation_proximity_buffer`).
    per_symbol_notional_pct: float = field(default_factory=lambda: _env_float("PER_SYMBOL_NOTIONAL_PCT", 0.30))
    liquidation_proximity_buffer: float = field(
        default_factory=lambda: _env_float("LIQUIDATION_PROXIMITY_BUFFER", 0.10)
    )
    # Vol-targeting у live/pairs_runner (2D): ERC + vol-target sizing замість
    # фіксованого position_pct. Дефолт False (зворотна сумісність); вмикати для
    # regime-scaled sizing прив'язаного до supervisor ваг.
    enable_vol_target: bool = field(default_factory=lambda: _env_bool("ENABLE_VOL_TARGET", False))
    vol_target_ann: float = field(default_factory=lambda: _env_float("VOL_TARGET_ANN", 0.10))
    max_losing_months: int = field(default_factory=lambda: _env_int("MAX_LOSING_MONTHS", 2))
    maker_fill_wait_bars: int = field(default_factory=lambda: _env_int("MAKER_FILL_WAIT_BARS", 1))
    # Hard-гейт overfitting-аудиту: live (DRY_RUN=false) і paper-run-pairs
    # вимагають свіжий PASS (пара — комірка LEG1/LEG2). Цей прапорець лишився
    # для інших paper-шляхів.
    require_audit_pass: bool = field(default_factory=lambda: _env_bool("REQUIRE_AUDIT_PASS", False))
    audit_max_age_days: int = field(default_factory=lambda: _env_int("AUDIT_MAX_AGE_DAYS", 30))
    # OOS-дисципліна (Narang гл. 9 — burning data): якщо True, audit_cell
    # перевіряє, чи OOS-вікно (strategy×symbol×дати) вже «спалене» в реєстрі, і
    # якщо так — повертає status=error (fail-closed). Після успішного аудиту —
    # автоматично дописує використання у реєстр. Дефолт False, щоб не ламати
    # повторні прогони дослідника; вмикати для дисциплінованого фінального аудиту.
    enforce_oos_burn: bool = field(default_factory=lambda: _env_bool("OOS_ENFORCE_BURN", False))
    oos_registry_path: Path = field(
        default_factory=lambda: Path(os.getenv("OOS_REGISTRY_PATH", "docs/reports/oos_usage.md"))
    )
    # «Замкований» holdout (Narang гл. 9): останні `holdout_pct`% даних НЕ
    # використовуються для підбору параметрів (WF/Optuna/sensitivity/DSR), а
    # лишаються для фінального сліпого тесту. Значення у відсотках 0–100
    # (напр. 20); 0 = вимкнено (дефолт, поточна поведінка). audit_cell за
    # замовчуванням тримає лише research-частину, якщо holdout_pct > 0.
    enforce_holdout_pct: float = field(default_factory=lambda: _env_float("HOLDOUT_PCT", 0.0))
    # Append-only журнал спроб (trials) для чесного DSR (замість «магічної» 50):
    # кожен бектест/аудит/оптимізація дописує рядок; лічильник дає реальну
    # кількість перебраних варіантів → n_trials для DSR. Порожнє → вимкнено.
    trial_ledger_path: Path = field(default_factory=lambda: Path(os.getenv("TRIAL_LEDGER_PATH", "")))

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
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    postgres_dsn: str = field(default_factory=lambda: os.getenv("POSTGRES_DSN", ""))

    # Telegram (опційно; без ключів send — no-op)
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    # Telegram Bot — whitelist chat_id через кому; порожнє → лише telegram_chat_id
    telegram_allowed_chat_ids: str = field(default_factory=lambda: os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""))
    # PIN для деструктивних команд (pause/stop/resume); порожнє → команди вимкнені
    telegram_bot_pin: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_PIN", ""))

    # Dashboard Auth
    dashboard_password_hash: str = field(default_factory=lambda: os.getenv("DASHBOARD_PASSWORD_HASH", ""))
    dashboard_host: str = field(default_factory=lambda: os.getenv("DASHBOARD_HOST", "127.0.0.1"))

    # API
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))
    api_secret_key: str = field(default_factory=lambda: os.getenv("API_SECRET_KEY", DEFAULT_API_SECRET_KEY))
    api_cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _parse_csv_tuple(os.getenv("API_CORS_ORIGINS", ""), _default_api_cors_origins())
    )
    api_allow_mock_positions: bool = field(default_factory=lambda: _env_bool("API_ALLOW_MOCK_POSITIONS", False))
    api_allow_destructive_ops: bool = field(default_factory=lambda: _env_bool("API_ALLOW_DESTRUCTIVE_OPS", False))

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

    @property
    def slippage(self) -> float:
        """Аліас для slippage_frac для сумісності з модулями бектесту."""
        return self.slippage_frac

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


def require_safe_api_bind(host: str, settings: Settings) -> None:
    """Fail-closed: слабкий JWT-секрет дозволений лише на localhost.

    Відхиляє на non-localhost бінді: порожній/дефолтний секрет, або коротший
    за 32 символи (JWT HS256 мінімум). У повідомленні немає значення ключа.
    """
    if host in _LOCAL_API_HOSTS:
        return
    key = settings.api_secret_key or ""
    if key == DEFAULT_API_SECRET_KEY or len(key) < 32:
        raise RuntimeError(
            f"API на {host} з порожнім/слабким API_SECRET_KEY заборонено. "
            "Задайте власний ключ (≥32 символи) у .env або біндіть на 127.0.0.1."
        )


def require_dashboard_password(host: str, settings: Settings) -> None:
    """Fail-closed: публічний bind дашборду без пароля заборонено."""
    bind = (host or settings.dashboard_host or "127.0.0.1").strip()
    if bind in _LOCAL_API_HOSTS:
        return
    pwd_hash = (settings.dashboard_password_hash or "").strip()
    if not pwd_hash:
        raise RuntimeError(
            f"Dashboard на {bind} без DASHBOARD_PASSWORD_HASH заборонено. "
            "Задайте bcrypt-хеш (cli dashboard-hash) або біндіть на 127.0.0.1."
        )


def set_settings(settings: Settings) -> None:
    """Оновити кешований синглтон Settings."""
    global _settings
    _settings = settings


def get_settings() -> Settings:
    """Кешований синглтон Settings (створюється один раз на процес).

    Streamlit тримає модулі між rerun-ами: після reload `config.py` клас
    `Settings` новий, а `_settings` може лишитися екземпляром старої версії.
    """
    global _settings
    if _settings is None or type(_settings) is not Settings:
        _settings = Settings()
        return _settings
    for name in Settings.__dataclass_fields__:
        if not hasattr(_settings, name):
            _settings = Settings()
            break
    return _settings
