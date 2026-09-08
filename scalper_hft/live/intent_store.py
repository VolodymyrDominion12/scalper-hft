"""Персистентний намір→clientOrderId (ідемпотентність між рестартами).

In-memory мапа `LiveTrader._intent_coids` гарантує retry з тим самим coid
лише в межах процесу. Після рестарту намір губився → можливий дубль ордера.
IntentStore тримає мапу у JSON (атомарний запис tmp+rename), тож retry
після краху процесу йде з тим самим coid, а біржа дедуплікує.

Формат файлу: {"<intent>": "<coid>", ...}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("results/intent_coids.json")


class IntentStore:
    """Потоко-небезпечність не критична: виклики з одного trader-циклу."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._map: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Битий intent store %s: %s — стартую з порожнього", self._path, exc)
        return {}

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._map, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)  # атомарно в межах однієї ФС

    def get(self, intent: str) -> str | None:
        return self._map.get(intent)

    def put(self, intent: str, coid: str) -> None:
        self._map[intent] = coid
        self._persist()

    def pop(self, intent: str) -> None:
        if intent in self._map:
            del self._map[intent]
            self._persist()

    def __len__(self) -> int:
        return len(self._map)
