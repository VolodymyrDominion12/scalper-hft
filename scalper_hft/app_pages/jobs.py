"""Сторінка «Задачі»: черга research-прогонів, логи, cancel / rerun."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore

st.title("Задачі")
st.caption("Черга бектестів і sweep. Дашборд лише ставить job; рахунок — у worker-процесі.")

store = JobStore(DEFAULT_JOBS_PATH)
alive = store.worker_is_alive()

with st.container(horizontal=True):
    if alive:
        st.badge("воркер активний", icon=":material/check_circle:", color="green")
    else:
        st.badge("воркер не запущений", icon=":material/pause_circle:", color="orange")
    st.caption("`uv run python -m scalper_hft.cli job worker --jobs 2`")

if not alive:
    st.info("Поставте job і запустіть воркер, або відкрийте дашборд без `--no-worker`.")


def _status_color(status: str) -> str:
    return {
        "queued": "blue",
        "running": "violet",
        "succeeded": "green",
        "failed": "red",
        "cancelled": "gray",
    }.get(status, "gray")


@st.fragment(run_every=2)
def _jobs_panel() -> None:
    with JobStore(DEFAULT_JOBS_PATH) as js:
        rows = js.list_jobs(limit=200)
        worker_on = js.worker_is_alive()
    if not worker_on:
        st.caption("воркер мовчить (немає heartbeat)")
    if not rows:
        st.info("Черга порожня.")
        return
    table = pd.DataFrame(
        [
            {
                "id": j.id,
                "kind": j.kind,
                "status": j.status,
                "progress": f"{j.progress_done}/{j.progress_total}" if j.progress_total else "—",
                "fp": j.short_fp,
                "error": (j.error or "")[:80],
                "created": j.created_at[11:19] if j.created_at else "",
            }
            for j in rows
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)
    ids = [j.id for j in rows]
    selected = st.selectbox("Деталі", ids, format_func=lambda i: f"#{i}", key="job_sel")
    job = next((j for j in rows if j.id == selected), None)
    if job is None:
        return
    st.badge(job.status, color=_status_color(job.status))
    st.json(job.params)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Скасувати", key="job_cancel", disabled=job.status not in {"queued", "running"}):
            with JobStore(DEFAULT_JOBS_PATH) as js:
                js.request_cancel(job.id)
            st.rerun()
    with c2:
        if st.button("Перезапустити", key="job_rerun"):
            with JobStore(DEFAULT_JOBS_PATH) as js:
                js.submit(job.kind, job.params, force=True)
            st.rerun()
    with JobStore(DEFAULT_JOBS_PATH) as js:
        log = js.tail_log(job.id, n=120)
    if log:
        st.code(log, language="text")
    else:
        st.caption("Логу ще немає.")


_jobs_panel()
store.close()
