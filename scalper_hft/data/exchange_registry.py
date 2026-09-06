"""Реєстр бірж для абстрагування специфіки (клас ccxt, дефолтні комісії, ліміти)."""

from __future__ import annotations

from dataclasses import dataclass

import ccxt


@dataclass(frozen=True)
class ExchangeMeta:
    """Метадані біржі для scalper-hft."""
    id: str
    ccxt_class: type[ccxt.Exchange]
    default_maker_fee: float
    default_taker_fee: float
    market_type: str = "future"
    sub_type: str = "linear"


class ExchangeRegistry:
    """Глобальний реєстр бірж."""
    
    _registry: dict[str, ExchangeMeta] = {
        "binance": ExchangeMeta(
            id="binance",
            ccxt_class=ccxt.binance,
            default_maker_fee=0.0002,
            default_taker_fee=0.0005,
        ),
        "binanceusdm": ExchangeMeta(
            id="binanceusdm",
            ccxt_class=ccxt.binanceusdm,
            default_maker_fee=0.0002,
            default_taker_fee=0.0005,
        ),
        "bybit": ExchangeMeta(
            id="bybit",
            ccxt_class=ccxt.bybit,
            default_maker_fee=0.0002,
            default_taker_fee=0.0005,
        ),
        "okx": ExchangeMeta(
            id="okx",
            ccxt_class=ccxt.okx,
            default_maker_fee=0.0002,
            default_taker_fee=0.0005,
        ),
    }

    @classmethod
    def get(cls, exchange_id: str) -> ExchangeMeta:
        """Отримати метадані біржі за ID."""
        clean_id = exchange_id.lower().strip()
        base_id = clean_id.replace("-testnet", "")
        if base_id not in cls._registry:
            # Fallback to general ccxt class if available, else binance
            if hasattr(ccxt, base_id):
                ccxt_cls = getattr(ccxt, base_id)
                return ExchangeMeta(
                    id=clean_id,
                    ccxt_class=ccxt_cls,
                    default_maker_fee=0.0002,
                    default_taker_fee=0.0005,
                )
            raise ValueError(f"Невідома біржа: {exchange_id}")
        
        # Make a copy for testnet if needed
        meta = cls._registry[base_id]
        if clean_id != base_id:
            return ExchangeMeta(
                id=clean_id,
                ccxt_class=meta.ccxt_class,
                default_maker_fee=meta.default_maker_fee,
                default_taker_fee=meta.default_taker_fee,
                market_type=meta.market_type,
                sub_type=meta.sub_type,
            )
        return meta

    @classmethod
    def list_supported(cls) -> list[str]:
        """Список підтримуваних бірж."""
        return list(cls._registry.keys())
