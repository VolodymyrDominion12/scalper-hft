"""Структурований збірник метрик live/paper (latency, WS, черги recorder)."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricsCollector:
    """Легкий in-process колектор: лічильники, gauge, латентності кроків."""

    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    gauges: dict[str, float] = field(default_factory=dict)
    latencies_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] += int(n)

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges[name] = float(value)

    def record_latency(self, name: str, ms: float) -> None:
        self.latencies_ms[name].append(float(ms))

    def record_step_latency(self, ms: float) -> None:
        """Латентність одного кроку trader/pairs loop."""
        self.record_latency("step_latency_ms", ms)

    def record_ws_reconnect(self) -> None:
        self.inc("ws_reconnects")

    def set_recorder_queue_depth(self, depth: int) -> None:
        self.set_gauge("recorder_queue_depth", float(depth))

    def snapshot(self) -> dict[str, Any]:
        lat_summary: dict[str, dict[str, float]] = {}
        for key, values in self.latencies_ms.items():
            if not values:
                continue
            sorted_v = sorted(values)
            n = len(sorted_v)
            lat_summary[key] = {
                "count": float(n),
                "p50": sorted_v[n // 2],
                "p95": sorted_v[int(n * 0.95)] if n > 1 else sorted_v[-1],
                "max": sorted_v[-1],
            }
        return {
            "ts": datetime.now(UTC).isoformat(),
            "uptime_sec": (datetime.now(UTC) - self.started_at).total_seconds(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "latencies": lat_summary,
        }

    def log_summary(self) -> None:
        snap = self.snapshot()
        logger.info(
            "metrics uptime=%.0fs counters=%s gauges=%s latencies=%s",
            snap["uptime_sec"],
            snap["counters"],
            snap["gauges"],
            snap["latencies"],
        )

    def export_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("metrics JSON: %s", p)


__all__ = ["MetricsCollector"]
