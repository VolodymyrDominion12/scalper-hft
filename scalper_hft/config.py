"""Конфігурація системи: змінні середовища з .env + розумні дефолти."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

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
    max_consecutive_losses: int = field(default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 3))

    # Дані
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    default_symbols: tuple[str, ...] = field(
        default_factory=lambda: tuple(s.strip() for s in os.getenv("DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip())
    )
    default_interval: str = field(default_factory=lambda: os.getenv("DEFAULT_INTERVAL", "1m"))

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


def get_settings() -> Settings:
    """Кешований синглтон Settings (створюється один раз на процес)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
