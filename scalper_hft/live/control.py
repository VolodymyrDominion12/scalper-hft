"""Control plane для paper/live демона: pause / no_new_entries / flatten.

Файл `control.json` читається на початку кожного `step`. Невалідний JSON
→ fail-closed (pause), щоб випадковий артефакт не лишив бота торгувати.
`flatten` діє лише якщо ключ явно true.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_PATH = Path("results") / "control.json"


@dataclass(frozen=True, slots=True)
class ControlState:
    pause: bool = False
    no_new_entries: bool = False
    flatten: bool = False

    @property
    def is_noop(self) -> bool:
        return not self.pause and not self.no_new_entries and not self.flatten


def load_control(path: Path | str | None = None) -> ControlState:
    """Прочитати control.json. Відсутній файл = дефолти (торгівля дозволена)."""
    p = Path(path) if path is not None else DEFAULT_CONTROL_PATH
    if not p.exists():
        return ControlState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("control.json нечитабельний (%s) — pause", exc)
        return ControlState(pause=True)
    if not isinstance(raw, dict):
        logger.error("control.json не об'єкт — pause")
        return ControlState(pause=True)
    return ControlState(
        pause=bool(raw.get("pause", False)),
        no_new_entries=bool(raw.get("no_new_entries", False)),
        flatten=raw.get("flatten") is True,
    )


def save_control(
    *,
    pause: bool | None = None,
    no_new_entries: bool | None = None,
    flatten: bool | None = None,
    path: Path | str | None = None,
) -> ControlState:
    """Атомарно оновити control.json і повернути актуальний стан."""
    p = Path(path) if path is not None else DEFAULT_CONTROL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if p.exists():
        try:
            current = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                current = {}
        except (OSError, json.JSONDecodeError):
            current = {}

    if pause is not None:
        current["pause"] = bool(pause)
    if no_new_entries is not None:
        current["no_new_entries"] = bool(no_new_entries)
    if flatten is not None:
        current["flatten"] = bool(flatten)

    # Атомарний запис через tmp файл
    tmp_file = p.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(current, indent=2), encoding="utf-8")
    tmp_file.replace(p)
    logger.info("control.json збережено (%s): %s", p, current)
    return load_control(p)
