"""Менеджер кешу для збереження даних по різних біржах."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scalper_hft.config import get_settings


class CacheManager:
    """Керує структурою директорій кешу та індексом (cache_index.json).
    
    Для кожної біржі створюється окрема піддиректорія (напр. data/binance/).
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or get_settings().data_dir_abs
        self.index_path = self.base_dir / "cache_index.json"
        self._ensure_base()
        self.index = self._load_index()

    def _ensure_base(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"version": 1, "exchanges": {}}
        try:
            with open(self.index_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"version": 1, "exchanges": {}}

    def _save_index(self) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def get_exchange_dir(self, exchange_id: str) -> Path:
        """Повертає шлях до директорії даних для вказаної біржі."""
        ex_dir = self.base_dir / exchange_id.lower().strip()
        ex_dir.mkdir(parents=True, exist_ok=True)
        return ex_dir

    def register_download(self, exchange_id: str, symbol: str, interval: str) -> None:
        """Оновлює індекс після завантаження даних."""
        ex = exchange_id.lower().strip()
        if ex not in self.index["exchanges"]:
            self.index["exchanges"][ex] = {}
        
        ex_data = self.index["exchanges"][ex]
        if symbol not in ex_data:
            ex_data[symbol] = []
            
        if interval not in ex_data[symbol]:
            ex_data[symbol].append(interval)
            
        self._save_index()

    def get_downloaded_symbols(self, exchange_id: str) -> list[str]:
        """Список символів, для яких є дані вказаної біржі."""
        ex = exchange_id.lower().strip()
        return list(self.index.get("exchanges", {}).get(ex, {}).keys())
