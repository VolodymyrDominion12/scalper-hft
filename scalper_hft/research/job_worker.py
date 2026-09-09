"""Воркер черги: claim → дочірній процес → heartbeat / SIGTERM cancel."""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from multiprocessing import Process, get_context
from pathlib import Path
from typing import Any, cast

from scalper_hft.research.jobs import (
    DEFAULT_JOBS_PATH,
    Job,
    JobStore,
    artifacts_dir,
    job_log_path,
)

logger = logging.getLogger(__name__)

POLL_SEC = 1.0
IDLE_SLEEP = 0.5


def _job_child_main(job_id: int, kind: str, params: dict, job_dir: str, store_path: str) -> None:
    """Точка входу дитини (picklable). Лог у job_dir/job.log."""
    log_path = Path(job_dir) / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )
    log = logging.getLogger("scalper_hft.job")
    log.info("start job id=%s kind=%s", job_id, kind)
    try:
        from scalper_hft.research.job_handlers import run_job

        run_job(kind, params, Path(job_dir), store_path=store_path, job_id=job_id)
    except Exception:
        log.exception("job id=%s failed", job_id)
        raise SystemExit(1) from None
    log.info("done job id=%s", job_id)


def _terminate(proc: Process) -> None:
    if not proc.is_alive() or proc.pid is None:
        return
    try:
        import psutil

        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        parent.terminate()
    except Exception:
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    proc.join(timeout=5.0)
    if proc.is_alive():
        try:
            import psutil

            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            parent.kill()
        except Exception:
            proc.kill()
        proc.join(timeout=2.0)


def run_claimed_job(store: JobStore, job: Job, *, mp_context: str | None = None) -> None:
    """Запустити вже claimed job у дочірньому процесі і дочекатися фіналу."""
    job_dir = artifacts_dir(store.path, job.id)
    job_dir.mkdir(parents=True, exist_ok=True)
    start_method = mp_context or "spawn"
    ctx = cast(Any, get_context(start_method))
    proc = ctx.Process(
        target=_job_child_main,
        args=(job.id, job.kind, job.params, str(job_dir), str(store.path)),
        name=f"scalper-job-{job.id}",
    )
    proc.start()
    if proc.pid is not None:
        store.heartbeat(job.id, pid=proc.pid)
    while proc.is_alive():
        store.heartbeat(job.id, pid=proc.pid)
        store.touch_worker()
        current = store.get(job.id)
        if current is not None and current.cancel_requested:
            logger.info("cancel job id=%s pid=%s", job.id, proc.pid)
            _terminate(proc)
            break
        proc.join(timeout=POLL_SEC)
    exitcode = proc.exitcode
    current = store.get(job.id)
    if current is None:
        return
    if current.force_requeue:
        store.requeue_after_cancel(job.id)
        logger.info("job id=%s force-requeued", job.id)
        return
    if current.cancel_requested or exitcode == -signal.SIGTERM:
        store.finish(current.id, "cancelled", error="cancelled", attempt=current.attempt)
        return
    if exitcode == 0:
        store.finish(current.id, "succeeded", attempt=current.attempt)
        return
    err = f"exit {exitcode}"
    logp = job_log_path(store.path, job.id)
    if logp.exists():
        tail = logp.read_text(encoding="utf-8", errors="replace").splitlines()
        if tail:
            err = tail[-1][:500]
    store.finish(current.id, "failed", error=err, attempt=current.attempt)


def worker_loop(
    store: JobStore,
    *,
    idle_sleep: float = IDLE_SLEEP,
    stop: list[bool] | None = None,
    mp_context: str | None = None,
) -> None:
    """Крутити claim→run, поки stop[0] не True (або вічно)."""
    logger.info("job worker pid=%s db=%s", os.getpid(), store.path)
    while stop is None or not stop[0]:
        store.touch_worker()
        job = store.claim(pid=os.getpid())
        if job is None:
            time.sleep(idle_sleep)
            continue
        logger.info("claimed job id=%s kind=%s fp=%s", job.id, job.kind, job.short_fp)
        try:
            run_claimed_job(store, job, mp_context=mp_context)
        except Exception:
            logger.exception("worker failed on job id=%s", job.id)
            store.finish(job.id, "failed", error="worker exception", attempt=job.attempt)


def spawn_workers(n: int, store_path: Path | str | None = None) -> None:
    """N процесів worker_loop у цьому інтерпретаторі (блокирує)."""
    path = Path(store_path) if store_path else DEFAULT_JOBS_PATH
    if n <= 1:
        with JobStore(path) as store:
            worker_loop(store)
        return
    ctx = get_context()
    procs: list[Process] = []

    def _one(p: str) -> None:
        with JobStore(p) as store:
            worker_loop(store)

    for i in range(n):
        proc = ctx.Process(target=_one, args=(str(path),), name=f"scalper-worker-{i}")
        proc.start()
        procs.append(proc)
        logger.info("started worker %s pid=%s", i, proc.pid)
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        for proc in procs:
            _terminate(proc)
