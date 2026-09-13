"""Відстеження ваги API-запитів Binance (`X-MBX-USED-WEIGHT-1M`).

Binance USDT-M futures обмежує REST-запити до 6000 одиниць ваги на хвилину за
IP-адресою. HTTP 429 — тимчасовий backoff, HTTP 418 — авто-бан IP на кілька
хвилин. Дослідження §7.2 («Стратегії MFT Криптоторгівлі 2026») наголошує:
MFT-системи мають програмно відстежувати заголовок `X-MBX-USED-WEIGHT-1M` і
вмикати exponential backoff при наближенні до ліміту.

Цей модуль — пасивний бюджет ваги: парсить заголовки після кожного ccxt-виклику
та рекомендує throttle, коли використання перевищує поріг (дефолт 92%).
Сам модуль **не** спить у тестах; рішення про sleep приймає викликач
(`ExchangeClient.throttle_if_needed` / `trader_loop`), щоб unit-тести не
блокувалися на `time.sleep`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Binance USDT-M futures: 6000 ваги/хв за IP (документація Binance 2026).
DEFAULT_WEIGHT_LIMIT = 6000
# Поріг у частці від ліміту, після якого рекомендуємо throttle.
DEFAULT_THROTTLE_PCT = 0.92
# Заголовки Binance, що несуть вагу (можливі варіанти імен).
_WEIGHT_HEADERS = (
    "X-MBX-USED-WEIGHT-1M",
    "X-MBX-USED-WEIGHT",
    "x-mbx-used-weight-1m",
    "x-mbx-used-weight",
)


@dataclass
class WeightBudget:
    """Пасивний лічильник ваги API-запитів Binance.

    Оновлюється з HTTP-заголовків після кожного запиту. Не робить мережних
    викликів сам; рішення про sleep делегує викликачу через `needs_throttle`.
    """

    weight_limit: int = DEFAULT_WEIGHT_LIMIT
    throttle_pct: float = DEFAULT_THROTTLE_PCT
    used_weight: int = 0
    # timestamp (ms) останнього оновлення — для визначення «свіжості» ваги.
    last_updated_ms: float = 0.0
    # Лічильник скільки разів рекомендували throttle (для метрик/дашборду).
    throttle_count: int = 0
    # Чи бачили ми хоч раз валідний заголовок (щоб не спати «наосліп»).
    _ever_seen_headers: bool = field(default=False, repr=False)

    @property
    def throttle_threshold(self) -> int:
        """Абсолютний поріг ваги, після якого рекомендуємо throttle."""
        return int(self.weight_limit * self.throttle_pct)

    @property
    def remaining(self) -> int:
        """Скільки ваги лишається до ліміту (≥0)."""
        return max(0, self.weight_limit - self.used_weight)

    @property
    def usage_pct(self) -> float:
        """Частка використання ліміту (0.0–1.0+)."""
        if self.weight_limit <= 0:
            return 0.0
        return self.used_weight / self.weight_limit

    @property
    def is_critical(self) -> bool:
        """Чи перевищено поріг throttle (рекомендуємо паузу перед наступним запитом)."""
        return self.used_weight >= self.throttle_threshold

    @property
    def is_banned(self) -> bool:
        """Чи вичерпано ліміт повністю (ризик HTTP 418 бана)."""
        return self.used_weight >= self.weight_limit

    @property
    def has_real_data(self) -> bool:
        """Чи бачили ми хоч раз валідний заголовок ваги.

        Викликачі (trader_loop) мають throttle лише коли це True, інакше
        спати «наосліп» без даних — безглуздо."""
        return self._ever_seen_headers

    def update_from_headers(self, headers: Mapping[str, Any] | None) -> None:
        """Оновити used_weight з HTTP-заголовків відповіді ccxt.

        ccxt зберігає заголовки останньої відповіді в
        `exchange.last_response_headers` (dict). Binance несе вагу в
        `X-MBX-USED-WEIGHT-1M` (цілочисельний рядок). Ігноруємо невалідні/
        відсутні заголовки тихо (біржа може не віддавати вагу на деяких
        ендпоінтах).
        """
        if not headers:
            return
        raw: Any = None
        for name in _WEIGHT_HEADERS:
            if name in headers:
                raw = headers[name]
                break
        if raw is None:
            return
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            logger.debug("Невалідний заголовок ваги: %r", raw)
            return
        if value < 0:
            return
        self.used_weight = value
        self.last_updated_ms = time.time() * 1000.0
        self._ever_seen_headers = True
        if self.is_critical:
            logger.warning(
                "API weight %d/%d (%.0f%%) ≥ поріг throttle %d",
                self.used_weight,
                self.weight_limit,
                self.usage_pct * 100.0,
                self.throttle_threshold,
            )

    def needs_throttle(self) -> bool:
        """Чи варто зробити паузу перед наступним запитом.

        True лише коли: (1) бачили валідні заголовки, (2) вага ≥ порогу.
        Викликач сам вирішує скільки спати (див. `recommended_sleep_s`).
        """
        return self.has_real_data and self.is_critical

    def recommended_sleep_s(self) -> float:
        """Рекомендована тривалість sleep (сек) при `needs_throttle`.

        Лінійна інтерполяція: на порозі throttle — коротка пауза (0.2 с),
        при наближенні до ліміту — довша (до 5 с). 0.0 якщо throttle не потрібен.

        Не блокує; викликач сам викликає `time.sleep(self.recommended_sleep_s())`.
        """
        if not self.needs_throttle():
            return 0.0
        # Від порогу (92%) до ліміту (100%) — лінійно 0.2 → 5.0 с.
        span = max(1, self.weight_limit - self.throttle_threshold)
        over = max(0, self.used_weight - self.throttle_threshold)
        frac = min(1.0, over / span)
        return 0.2 + frac * 4.8

    def reset(self) -> None:
        """Скинути лічильник (наприклад, після хвилинного вікна Binance)."""
        self.used_weight = 0
        self.last_updated_ms = 0.0
        self.throttle_count = 0

    def mark_throttle(self) -> None:
        """Викликач позначає, що зробив паузу (для метрик)."""
        self.throttle_count += 1

    def snapshot(self) -> dict[str, Any]:
        """Стан для логів/дашборду (без мережі)."""
        return {
            "used_weight": self.used_weight,
            "weight_limit": self.weight_limit,
            "usage_pct": round(self.usage_pct * 100.0, 1),
            "remaining": self.remaining,
            "is_critical": self.is_critical,
            "is_banned": self.is_banned,
            "has_real_data": self.has_real_data,
            "throttle_count": self.throttle_count,
            "last_updated_ms": self.last_updated_ms,
        }
