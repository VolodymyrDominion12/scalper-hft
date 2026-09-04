"""Черга дослідницьких задач: SQLite, fingerprint, ідемпотентний submit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_JOBS_PATH = Path("results") / "jobs.sqlite"
STALE_AFTER_SEC = 30.0
WORKER_ALIVE_SEC = 15.0
_SORT_LIST_KEYS = frozenset({"strategies", "symbols", "intervals"})

_CREATE = """
CREATE TABLE IF NOT EXISTS jobs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint        TEXT NOT NULL UNIQUE,
    kind               TEXT NOT NULL,
    params_json        TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'queued',
    progress_done      INTEGER NOT NULL DEFAULT 0,
    progress_total     INTEGER NOT NULL DEFAULT 0,
    pid                INTEGER,
    heartbeat_at       TEXT NOT NULL DEFAULT '',
    created_at         TEXT NOT NULL,
    started_at         TEXT NOT NULL DEFAULT '',
    finished_at        TEXT NOT NULL DEFAULT '',
    error              TEXT NOT NULL DEFAULT '',
    attempt            INTEGER NOT NULL DEFAULT 0,
    cancel_requested   INTEGER NOT NULL DEFAULT 0,
    force_requeue      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def canonicalize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Стабільне дерево params: відсортовані ключі, списки стратегій/символів/TF."""
    return _canon(dict(params))  # type: ignore[arg-type]


def _canon(obj: Any, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canon(v, str(k)) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        items = [_canon(x) for x in obj]
        if key in _SORT_LIST_KEYS:
            try:
                return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, default=str))
            except TypeError:
                return items
        return items
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, str):
        return obj
    return str(obj)


def fingerprint(kind: str, params: Mapping[str, Any]) -> str:
    """SHA-256 від kind + канонічних params (без timestamp)."""
    payload = {"kind": kind, "params": canonicalize_params(params)}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Job:
    id: int
    fingerprint: str
    kind: str
    params: dict[str, Any]
    status: str
    progress_done: int = 0
    progress_total: int = 0
    pid: int | None = None
    heartbeat_at: str = ""
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    attempt: int = 0
    cancel_requested: bool = False
    force_requeue: bool = False

    @property
    def short_fp(self) -> str:
        return self.fingerprint[:12]


@dataclass(frozen=True, slots=True)
class PruneStats:
    scanned_jobs: int
    pruned_jobs: int
    deleted_dirs: int
    freed_bytes: int
    orphaned_dirs: int

    @property
    def freed_mb(self) -> float:
        return self.freed_bytes / (1024 * 1024)


def artifacts_dir(store_path: Path, job_id: int) -> Path:
    """Каталог артефактів: <parent>/jobs/<id>/ поруч із jobs.sqlite."""
    return Path(store_path).parent / "jobs" / str(job_id)


def job_log_path(store_path: Path, job_id: int) -> Path:
    return artifacts_dir(store_path, job_id) / "job.log"


class JobStore:
    """WAL SQLite черга. Одна job на fingerprint; submit ідемпотентний."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_JOBS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None  # autocommit; явні BEGIN IMMEDIATE
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE)
        self._conn.execute("PRAGMA busy_timeout=5000")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> JobStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        params = json.loads(row["params_json"] or "{}")
        if not isinstance(params, dict):
            params = {}
        pid = row["pid"]
        return Job(
            id=int(row["id"]),
            fingerprint=str(row["fingerprint"]),
            kind=str(row["kind"]),
            params=params,
            status=str(row["status"]),
            progress_done=int(row["progress_done"] or 0),
            progress_total=int(row["progress_total"] or 0),
            pid=int(pid) if pid is not None else None,
            heartbeat_at=str(row["heartbeat_at"] or ""),
            created_at=str(row["created_at"] or ""),
            started_at=str(row["started_at"] or ""),
            finished_at=str(row["finished_at"] or ""),
            error=str(row["error"] or ""),
            attempt=int(row["attempt"] or 0),
            cancel_requested=bool(row["cancel_requested"]),
            force_requeue=bool(row["force_requeue"]),
        )

    def get(self, job_id: int) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def get_by_fingerprint(self, fp: str) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE fingerprint=?", (fp,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, *, limit: int = 100) -> list[Job]:
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def submit(self, kind: str, params: Mapping[str, Any], *, force: bool = False) -> Job:
        """Поставити job. Без force: succeeded/queued/running не дублюються."""
        canon = canonicalize_params(params)
        fp = fingerprint(kind, canon)
        now = _now()
        blob = json.dumps(canon, sort_keys=True, ensure_ascii=True)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT * FROM jobs WHERE fingerprint=?", (fp,)).fetchone()
                if row is None:
                    cur = self._conn.execute(
                        """INSERT INTO jobs (fingerprint, kind, params_json, status, created_at, attempt)
                           VALUES (?, ?, ?, 'queued', ?, 0)""",
                        (fp, kind, blob, now),
                    )
                    if cur.lastrowid is None:
                        raise RuntimeError("Не вдалося отримати lastrowid для нового job")
                    job_id = int(cur.lastrowid)
                    self._conn.execute("COMMIT")
                    job = self.get(job_id)
                    assert job is not None
                    return job
                existing = self._row_to_job(row)
                if force:
                    self._force_unlocked(existing)
                    self._conn.execute("COMMIT")
                    job = self.get(existing.id)
                    assert job is not None
                    return job
                if existing.status in {"queued", "running", "succeeded"}:
                    self._conn.execute("COMMIT")
                    return existing
                self._requeue_unlocked(existing.id, cancel=False)
                self._conn.execute("COMMIT")
                job = self.get(existing.id)
                assert job is not None
                return job
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _force_unlocked(self, existing: Job) -> None:
        self._clear_artifacts(existing.id)
        if existing.status == "running":
            self._conn.execute(
                """UPDATE jobs SET cancel_requested=1, force_requeue=1, error='',
                   progress_done=0, progress_total=0 WHERE id=?""",
                (existing.id,),
            )
            return
        self._requeue_unlocked(existing.id, cancel=False)

    def _requeue_unlocked(self, job_id: int, *, cancel: bool) -> None:
        self._conn.execute(
            """UPDATE jobs SET status='queued', pid=NULL, heartbeat_at='', started_at='',
               finished_at='', error='', progress_done=0, progress_total=0,
               cancel_requested=?, force_requeue=0, attempt=attempt+1 WHERE id=?""",
            (1 if cancel else 0, job_id),
        )

    def _clear_artifacts(self, job_id: int) -> None:
        path = artifacts_dir(self.path, job_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def claim(self, *, pid: int | None = None) -> Job | None:
        """Атомарно взяти найстарішу queued job. None якщо черга порожня."""
        self.reap_stale()
        pid = pid if pid is not None else os.getpid()
        now = _now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at ASC, id ASC LIMIT 1"
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None
                job_id = int(row["id"])
                cur = self._conn.execute(
                    """UPDATE jobs SET status='running', pid=?, heartbeat_at=?, started_at=?,
                       finished_at='', error='', cancel_requested=0, force_requeue=0
                       WHERE id=? AND status='queued'""",
                    (pid, now, now, job_id),
                )
                self._conn.execute("COMMIT")
                if cur.rowcount != 1:
                    return None
                return self.get(job_id)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def heartbeat(self, job_id: int, *, pid: int | None = None) -> None:
        now = _now()
        if pid is not None:
            self._conn.execute(
                "UPDATE jobs SET heartbeat_at=?, pid=? WHERE id=? AND status='running'",
                (now, pid, job_id),
            )
        else:
            self._conn.execute(
                "UPDATE jobs SET heartbeat_at=? WHERE id=? AND status='running'",
                (now, job_id),
            )

    def set_progress(self, job_id: int, done: int, total: int) -> None:
        self._conn.execute(
            "UPDATE jobs SET progress_done=?, progress_total=? WHERE id=?",
            (done, total, job_id),
        )

    def request_cancel(self, job_id: int) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job.status == "queued":
            self._conn.execute(
                """UPDATE jobs SET status='cancelled', cancel_requested=1, finished_at=?
                   WHERE id=? AND status='queued'""",
                (_now(), job_id),
            )
        elif job.status == "running":
            self._conn.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
        return self.get(job_id)

    def finish(
        self,
        job_id: int,
        status: str,
        *,
        error: str = "",
        attempt: int | None = None,
    ) -> bool:
        """Закрити running-job. False якщо статус/attempt уже змінені (force requeue)."""
        now = _now()
        if attempt is None:
            cur = self._conn.execute(
                """UPDATE jobs SET status=?, error=?, finished_at=?, pid=NULL,
                   cancel_requested=0, force_requeue=0
                   WHERE id=? AND status='running'""",
                (status, error, now, job_id),
            )
        else:
            cur = self._conn.execute(
                """UPDATE jobs SET status=?, error=?, finished_at=?, pid=NULL,
                   cancel_requested=0, force_requeue=0
                   WHERE id=? AND status='running' AND attempt=?""",
                (status, error, now, job_id, attempt),
            )
        return cur.rowcount == 1

    def requeue_after_cancel(self, job_id: int) -> None:
        """Після смерті дитини, якщо force_requeue: знову queued."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return
                job = self._row_to_job(row)
                if job.force_requeue:
                    self._requeue_unlocked(job_id, cancel=False)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def reap_stale(self, *, max_age_sec: float = STALE_AFTER_SEC) -> int:
        """running без heartbeat → queued (воркер помер)."""
        cutoff = datetime.now(UTC) - timedelta(seconds=max_age_sec)
        rows = self._conn.execute("SELECT id, heartbeat_at FROM jobs WHERE status='running'").fetchall()
        n = 0
        for row in rows:
            hb = parse_iso(str(row["heartbeat_at"] or ""))
            if hb is None or hb < cutoff:
                self._conn.execute(
                    """UPDATE jobs SET status='queued', pid=NULL, error='stale heartbeat',
                       cancel_requested=0, force_requeue=0, attempt=attempt+1
                       WHERE id=? AND status='running'""",
                    (int(row["id"]),),
                )
                n += 1
        return n

    def touch_worker(self) -> None:
        self._conn.execute(
            "INSERT INTO meta (k, v) VALUES ('worker_heartbeat', ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (_now(),),
        )

    def worker_is_alive(self, *, max_age_sec: float = WORKER_ALIVE_SEC) -> bool:
        row = self._conn.execute("SELECT v FROM meta WHERE k='worker_heartbeat'").fetchone()
        if row is None:
            return False
        ts = parse_iso(str(row["v"]))
        if ts is None:
            return False
        return datetime.now(UTC) - ts <= timedelta(seconds=max_age_sec)

    def tail_log(self, job_id: int, n: int = 80) -> str:
        path = job_log_path(self.path, job_id)
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])

    def prune_jobs(
        self,
        days: int = 14,
        status: str = "all",
        keep_records: bool = False,
        dry_run: bool = False,
    ) -> PruneStats:
        """Очистити застарілі завершені задачі та їхні артефакти на диску.

        Ніколи не видаляє задачі зі статусом queued або running.
        """
        from scalper_hft.research.job_artifacts import calculate_job_artifacts_size

        terminal_statuses = {"succeeded", "failed", "cancelled"}
        if status == "all":
            allowed_statuses = tuple(terminal_statuses)
        elif status in terminal_statuses:
            allowed_statuses = (status,)
        else:
            raise ValueError(
                f"Неприпустимий статус для prune: {status}. Дозволені: all, {', '.join(sorted(terminal_statuses))}"
            )

        cutoff = datetime.now(UTC) - timedelta(days=max(0, int(days)))

        placeholders = ",".join("?" * len(allowed_statuses))
        rows = self._conn.execute(
            f"SELECT id, status, created_at, finished_at FROM jobs WHERE status IN ({placeholders})",
            allowed_statuses,
        ).fetchall()

        scanned = len(rows)
        pruned_jobs = 0
        deleted_dirs = 0
        freed_bytes = 0
        orphaned_dirs = 0
        deleted_ids: list[int] = []

        for row in rows:
            job_id = int(row["id"])
            ts = parse_iso(str(row["finished_at"] or "")) or parse_iso(str(row["created_at"] or ""))
            if ts is not None and ts < cutoff:
                pruned_jobs += 1
                art_dir = artifacts_dir(self.path, job_id)
                if art_dir.exists() and art_dir.is_dir():
                    size = calculate_job_artifacts_size(art_dir)
                    freed_bytes += size
                    deleted_dirs += 1
                    if not dry_run:
                        shutil.rmtree(art_dir, ignore_errors=True)
                if not keep_records:
                    deleted_ids.append(job_id)

        if deleted_ids and not dry_run:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    for jid in deleted_ids:
                        self._conn.execute("DELETE FROM jobs WHERE id=?", (jid,))
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise

        # Пошук та очищення orphaned-каталогів у <parent>/jobs/
        base_jobs_dir = Path(self.path).parent / "jobs"
        if base_jobs_dir.exists() and base_jobs_dir.is_dir():
            existing_ids = {int(r["id"]) for r in self._conn.execute("SELECT id FROM jobs").fetchall()}
            for child in base_jobs_dir.iterdir():
                if child.is_dir() and child.name.isdigit():
                    cid = int(child.name)
                    if cid not in existing_ids:
                        size = calculate_job_artifacts_size(child)
                        freed_bytes += size
                        deleted_dirs += 1
                        orphaned_dirs += 1
                        if not dry_run:
                            shutil.rmtree(child, ignore_errors=True)

        return PruneStats(
            scanned_jobs=scanned,
            pruned_jobs=pruned_jobs,
            deleted_dirs=deleted_dirs,
            freed_bytes=freed_bytes,
            orphaned_dirs=orphaned_dirs,
        )

