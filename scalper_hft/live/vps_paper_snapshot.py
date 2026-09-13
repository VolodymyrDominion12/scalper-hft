"""Знімок VPS paper-ботів для локального моніторингу (тільки читання).

`scripts/sync_vps_paper.sh` привозить у `results/vps/` консистентні копії
paper-журналів (sqlite backup API, без рваного WAL) разом із `manifest.json`,
у якому є стан systemd-юнітів і хеші файлів. Цей модуль читає копії і **нічого
не пише**: SQLite відкривається через `mode=ro` URI, а не через `PaperStore`
(його конструктор створює та мігрує схему, тобто змінював би знімок).

Навіщо окремий шлях: журнали живуть на VPS, а аналітика й дашборд — локально
(`docs/RUNBOOK_PAPER_MONITORING.md`, варіант «snapshot sync»). Свіжість
вимірюється двома незалежними сигналами — час останнього запису в журнал і
стан юніта з маніфесту, — бо tsmom пише equity лише на закритті бару
(1d → до 24 год тиші при цілком здоровому процесі).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SNAPSHOT_DIR = Path("results") / "vps"

#: Запас до інтервалу бару: loop спить до години і пише equity на кроці,
#: тому вік журналу в межах interval + 1 год — норма.
MAX_AGE_SLACK_HOURS = 2.0

#: Скільки може жити сам знімок, перш ніж вважати синк зламаним.
DEFAULT_MAX_SYNC_AGE_HOURS = 2.0


@dataclass(frozen=True)
class BotSpec:
    """Один paper-бот: журнал у знімку + відповідний systemd-юніт на VPS."""

    key: str
    label: str
    db: str
    unit: str
    interval: str
    interval_hours: float
    portfolio_id: str

    @property
    def max_age_hours(self) -> float:
        return self.interval_hours + MAX_AGE_SLACK_HOURS


BOTS: tuple[BotSpec, ...] = (
    BotSpec(
        key="pairs_1h",
        label="pairs_arb LINK/BTC 1h maker",
        db="paper_pairs.sqlite",
        unit="scalper-paper-pairs",
        interval="1h",
        interval_hours=1.0,
        portfolio_id="LINKUSDT/BTCUSDT",
    ),
    BotSpec(
        key="tsmom_1d",
        label="ts_momentum 1d long-only",
        db="paper_ts_momentum_1d.sqlite",
        unit="scalper-paper-tsmom@1d",
        interval="1d",
        interval_hours=24.0,
        portfolio_id="TSMOM_PORTFOLIO",
    ),
    BotSpec(
        key="tsmom_4h",
        label="ts_momentum 4h long-only",
        db="paper_ts_momentum_4h.sqlite",
        unit="scalper-paper-tsmom@4h",
        interval="4h",
        interval_hours=4.0,
        portfolio_id="TSMOM_PORTFOLIO",
    ),
)

BOTS_BY_KEY: dict[str, BotSpec] = {spec.key: spec for spec in BOTS}


def parse_ts(value: Any) -> datetime | None:
    """Розібрати позначку часу з журналу. Naive-значення трактуємо як UTC."""
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace(" ", "T"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass
class JournalStats:
    """Те, що читається з одного paper-журналу (усі поля — best effort)."""

    exists: bool = False
    equity_rows: int = 0
    trades: int = 0
    open_positions: int = 0
    orders: int = 0
    orders_by_status: dict[str, int] = field(default_factory=dict)
    symbols: int = 0
    equity: float | None = None
    realized_pnl: float | None = None
    last_equity_ts: datetime | None = None
    last_snapshot_ts: datetime | None = None

    @property
    def last_write_ts(self) -> datetime | None:
        """Найсвіжіша позначка запису: equity (крок) або runtime snapshot."""
        candidates = [ts for ts in (self.last_equity_ts, self.last_snapshot_ts) if ts is not None]
        return max(candidates) if candidates else None


def _ro_connect(path: Path) -> sqlite3.Connection:
    """Відкрити копію журналу тільки на читання.

    Якщо поруч немає непорожнього `-wal`, додаємо `immutable=1`: копія статична
    (синк підміняє файл через `os.replace`), тому локи не потрібні, а SQLite не
    створює в каталозі знімка порожні `-wal`/`-shm`. Якщо `-wal` таки є (тести
    або ручна копія з-під живого процесу), читаємо звичайним `mode=ro`, щоб не
    втратити незачекпойнчені дані.
    """
    wal = path.with_name(path.name + "-wal")
    has_pending_wal = wal.exists() and wal.stat().st_size > 0
    uri = f"file:{path}?mode=ro" if has_pending_wal else f"file:{path}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def _scalar(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    try:
        row = con.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else row[0]


def read_journal(path: Path, spec: BotSpec) -> JournalStats:
    """Прочитати зведення по одному журналу. Відсутні таблиці — не помилка."""
    stats = JournalStats(exists=path.exists())
    if not stats.exists:
        return stats

    con = _ro_connect(path)
    try:
        stats.equity_rows = int(_scalar(con, "SELECT COUNT(*) FROM equity") or 0)
        stats.trades = int(_scalar(con, "SELECT COUNT(*) FROM trades") or 0)
        stats.open_positions = int(_scalar(con, "SELECT COUNT(*) FROM positions") or 0)
        stats.orders = int(_scalar(con, "SELECT COUNT(*) FROM orders") or 0)
        stats.symbols = int(_scalar(con, "SELECT COUNT(DISTINCT pair) FROM equity") or 0)
        stats.last_equity_ts = parse_ts(_scalar(con, "SELECT MAX(ts) FROM equity"))
        stats.last_snapshot_ts = parse_ts(_scalar(con, "SELECT MAX(saved_at) FROM snapshots"))

        try:
            rows = con.execute("SELECT status, COUNT(*) FROM orders GROUP BY status").fetchall()
            stats.orders_by_status = {str(r[0]): int(r[1]) for r in rows}
        except sqlite3.Error:
            stats.orders_by_status = {}

        last_ts = _scalar(con, "SELECT MAX(ts) FROM equity WHERE pair = ?", (spec.portfolio_id,))
        if last_ts is not None:
            row = con.execute(
                "SELECT equity, realized_pnl FROM equity WHERE pair = ? AND ts = ? ORDER BY id DESC LIMIT 1",
                (spec.portfolio_id, last_ts),
            ).fetchone()
            if row is not None:
                stats.equity = float(row["equity"])
                stats.realized_pnl = float(row["realized_pnl"] or 0.0)
                return stats

        # Портфельного рядка немає (старіші журнали) — сумуємо символи на останньому кроці.
        if stats.last_equity_ts is not None:
            row = con.execute(
                "SELECT SUM(equity) AS eq, SUM(realized_pnl) AS pnl FROM equity WHERE ts = (SELECT MAX(ts) FROM equity)"
            ).fetchone()
            if row is not None and row["eq"] is not None:
                stats.equity = float(row["eq"])
                stats.realized_pnl = float(row["pnl"] or 0.0)
    finally:
        con.close()
    return stats


@dataclass
class BotSnapshot:
    """Стан одного бота за локальною копією + юніт із маніфесту."""

    spec: BotSpec
    db_path: Path
    journal: JournalStats
    db_bytes: int | None = None
    db_sha256: str = ""
    unit_active: str = "unknown"
    unit_since: str = ""
    unit_last_log_ts: datetime | None = None
    unit_last_log: str = ""
    issues: list[str] = field(default_factory=list)
    #: Час «зараз» для розрахунку віку; інжектується в тестах.
    _now: datetime | None = field(default=None, repr=False, compare=False)

    @property
    def last_write_ts(self) -> datetime | None:
        return self.journal.last_write_ts

    @property
    def age_hours(self) -> float | None:
        return None if self._now is None or self.last_write_ts is None else _hours(self.last_write_ts, self._now)

    @property
    def healthy(self) -> bool:
        return not self.issues


@dataclass
class SnapshotReport:
    """Повний звіт: знімок загалом + кожен бот + стан control.json."""

    snapshot_dir: Path
    snapshots: list[BotSnapshot]
    manifest: dict[str, Any] = field(default_factory=dict)
    control: dict[str, Any] = field(default_factory=dict)
    synced_at: datetime | None = None
    sync_age_hours: float | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.issues and all(s.healthy for s in self.snapshots)

    def by_key(self, key: str) -> BotSnapshot | None:
        return next((s for s in self.snapshots if s.spec.key == key), None)


def _hours(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def load_manifest(snapshot_dir: Path) -> dict[str, Any]:
    path = snapshot_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def load_snapshot_control(snapshot_dir: Path) -> dict[str, Any]:
    """Копія `results/control.json` з VPS (спільний стоп-кран трьох ботів)."""
    path = snapshot_dir / "control.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def build_report(
    snapshot_dir: Path | str = DEFAULT_SNAPSHOT_DIR,
    *,
    now: datetime | None = None,
    max_sync_age_hours: float = DEFAULT_MAX_SYNC_AGE_HOURS,
    bots: tuple[BotSpec, ...] = BOTS,
) -> SnapshotReport:
    """Зібрати звіт по знімку. `now` інжектується для тестів."""
    now = now or datetime.now(UTC)
    snapshot_dir = Path(snapshot_dir)
    manifest = load_manifest(snapshot_dir)
    dbs_meta: dict[str, Any] = manifest.get("databases", {}) if isinstance(manifest, dict) else {}
    units_meta: dict[str, Any] = manifest.get("units", {}) if isinstance(manifest, dict) else {}
    synced_at = parse_ts(manifest.get("synced_at_local") or manifest.get("remote_time"))

    issues: list[str] = []
    if not snapshot_dir.exists():
        issues.append(f"немає каталогу знімка {snapshot_dir} — запустіть scripts/sync_vps_paper.sh")
    elif not manifest:
        issues.append(f"немає {snapshot_dir}/manifest.json — синк не виконувався")

    age_hours = None if synced_at is None else _hours(synced_at, now)
    if age_hours is not None and age_hours > max_sync_age_hours:
        issues.append(f"знімок застарілий: {age_hours:.2f} год > {max_sync_age_hours:.2f} год")

    snapshots: list[BotSnapshot] = []
    for spec in bots:
        db_path = snapshot_dir / spec.db
        journal = read_journal(db_path, spec)
        meta = dbs_meta.get(spec.db, {}) if isinstance(dbs_meta, dict) else {}
        unit = units_meta.get(spec.unit, {}) if isinstance(units_meta, dict) else {}
        snap = BotSnapshot(
            spec=spec,
            db_path=db_path,
            journal=journal,
            db_bytes=meta.get("bytes"),
            db_sha256=str(meta.get("sha256", "")),
            unit_active=str(unit.get("active", "unknown")),
            unit_since=str(unit.get("since", "") or ""),
            unit_last_log_ts=parse_ts(unit.get("last_log_ts")),
            unit_last_log=str(unit.get("last_log", "") or ""),
            _now=now,
        )
        if not snap.journal.exists:
            snap.issues.append("немає копії журналу")
        if snap.unit_active not in ("active", "unknown"):
            snap.issues.append(f"юніт на VPS: {snap.unit_active}")
        last_ts = snap.last_write_ts
        if last_ts is None:
            snap.issues.append("у журналі немає жодного запису equity")
        elif age_hours is not None and _hours(last_ts, now) > spec.max_age_hours:
            snap.issues.append(
                f"немає записів {_hours(last_ts, now):.1f} год > {spec.max_age_hours:.1f} год "
                f"(інтервал {spec.interval})"
            )
        snapshots.append(snap)

    return SnapshotReport(
        snapshot_dir=snapshot_dir,
        snapshots=snapshots,
        manifest=manifest,
        control=load_snapshot_control(snapshot_dir),
        synced_at=synced_at,
        sync_age_hours=age_hours,
        issues=issues,
    )


def read_equity_series(path: Path, spec: BotSpec, *, limit: int = 500) -> list[tuple[datetime, float]]:
    """Історія equity портфеля з копії журналу (зростання за часом)."""
    if not path.exists():
        return []
    con = _ro_connect(path)
    try:
        rows = con.execute(
            "SELECT ts, equity FROM equity WHERE pair = ? ORDER BY id DESC LIMIT ?",
            (spec.portfolio_id, limit),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()

    series: list[tuple[datetime, float]] = []
    for row in reversed(rows):
        ts = parse_ts(row["ts"])
        if ts is None:
            continue
        series.append((ts, float(row["equity"])))
    return series


def read_trades(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    """Останні угоди з копії журналу (для таблиці на дашборді)."""
    if not path.exists():
        return []
    con = _ro_connect(path)
    try:
        rows = con.execute(
            "SELECT ts, pair, symbol, side, size, entry_price, exit_price, pnl, kind "
            "FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [dict(row) for row in rows]


def pairs_gate_status(snapshot_dir: Path | str = DEFAULT_SNAPSHOT_DIR) -> tuple[bool, str]:
    """Стан hard-гейта overfitting-аудиту для pairs-комірки (як на VPS)."""
    from scalper_hft.live.audit_gate import audit_gate_check_pair

    path = Path(snapshot_dir) / "audit_verdicts.jsonl"
    if not path.exists():
        return False, f"немає {path} — вердикти не синхронізовані"
    return audit_gate_check_pair("pairs_arb", "LINKUSDT", "BTCUSDT", "1h", path=path)


def format_age(hours: float | None) -> str:
    if hours is None:
        return "—"
    if hours < 1.0:
        return f"{hours * 60:.0f} хв"
    if hours < 48.0:
        return f"{hours:.1f} год"
    return f"{hours / 24:.1f} діб"


def format_table(report: SnapshotReport) -> str:
    """Текстовий статус для CLI (`scripts/vps_paper_status.py`)."""
    lines: list[str] = []
    sync_state = "—" if report.synced_at is None else report.synced_at.isoformat(timespec="seconds")
    lines.append(f"Знімок: {report.snapshot_dir} | синк: {sync_state} ({format_age(report.sync_age_hours)} тому)")
    lines.append("")
    lines.append(f"{'Бот':<32} {'Юніт':<11} {'Журнал':<12} {'Equity':>12} {'PnL':>9} {'Угод':>6} {'Поз':>4}")
    for snap in report.snapshots:
        stats = snap.journal
        equity = "—" if stats.equity is None else f"{stats.equity:,.2f}"
        pnl = "—" if stats.realized_pnl is None else f"{stats.realized_pnl:+,.2f}"
        lines.append(
            f"{snap.spec.label:<32} {snap.unit_active:<11} {format_age(snap.age_hours):<12} "
            f"{equity:>12} {pnl:>9} {stats.trades:>6} {stats.open_positions:>4}"
        )
    lines.append("")

    control = report.control or {}
    if control:
        flags = ", ".join(f"{k}={v}" for k, v in control.items())
        lines.append(f"control.json на VPS: {flags}")
    else:
        lines.append("control.json на VPS: не синхронізовано")

    problems = list(report.issues)
    for snap in report.snapshots:
        problems.extend(f"{snap.spec.label}: {msg}" for msg in snap.issues)
    if problems:
        lines.append("Проблеми:")
        lines.extend(f"  ✗ {msg}" for msg in problems)
    else:
        lines.append("Проблеми: немає — усі три боти здорові")
    return "\n".join(lines)


__all__ = [
    "BOTS",
    "BOTS_BY_KEY",
    "DEFAULT_MAX_SYNC_AGE_HOURS",
    "DEFAULT_SNAPSHOT_DIR",
    "BotSnapshot",
    "BotSpec",
    "JournalStats",
    "SnapshotReport",
    "build_report",
    "format_age",
    "format_table",
    "load_manifest",
    "load_snapshot_control",
    "pairs_gate_status",
    "parse_ts",
    "read_equity_series",
    "read_journal",
    "read_trades",
]
