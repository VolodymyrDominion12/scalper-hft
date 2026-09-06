"""Сторінка «Задачі»: черга research-прогонів, логи, cancel / rerun."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st
from scalper_hft.app_pages._common import (
    JOB_KINDS,
    JOB_STATUSES,
    job_label,
    job_open_target,
)
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
    st.page_link("app_pages/help.py", label="Довідка", icon=":material/help:")


def _status_color(
    status: str,
) -> Literal["red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"]:
    colors: dict[str, Literal["red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"]] = {
        "queued": "blue",
        "running": "violet",
        "succeeded": "green",
        "failed": "red",
        "cancelled": "gray",
    }
    return colors.get(status, "gray")


@st.fragment(run_every=2)
def _jobs_panel() -> None:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        status_pick = st.pills("Статус", ["усі", *JOB_STATUSES], default="усі", key="job_status_filter")
        kind_pick = st.pills("Тип", ["усі", *JOB_KINDS], default="усі", key="job_kind_filter")
    status_arg = None if not status_pick or status_pick == "усі" else status_pick
    kind_arg = None if not kind_pick or kind_pick == "усі" else kind_pick
    with JobStore(DEFAULT_JOBS_PATH) as js:
        rows = js.list_jobs(limit=200, status=status_arg, kind=kind_arg)
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
                "підпис": job_label(j.kind, j.params),
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
    labels = {j.id: f"#{j.id} · {job_label(j.kind, j.params)}" for j in rows}
    selected = st.selectbox(
        "Деталі",
        ids,
        format_func=lambda i: labels.get(i, f"#{i}"),
        key="job_sel",
    )
    job = next((j for j in rows if j.id == selected), None)
    if job is None:
        return
    st.badge(job.status, color=_status_color(job.status))
    st.caption(job_label(job.kind, job.params))
    if job.progress_total:
        st.progress(min(1.0, job.progress_done / job.progress_total))
    with st.expander("Параметри", icon=":material/data_object:"):
        st.json(job.params)
    with st.container(horizontal=True):
        if st.button(
            "Скасувати", icon=":material/cancel:", key="job_cancel", disabled=job.status not in {"queued", "running"}
        ):
            with JobStore(DEFAULT_JOBS_PATH) as js:
                js.request_cancel(job.id)
            st.rerun()
        if st.button("Перезапустити", icon=":material/replay:", key="job_rerun"):
            with JobStore(DEFAULT_JOBS_PATH) as js:
                js.submit(job.kind, job.params, force=True)
            st.rerun()
        page, updates = job_open_target(job.kind, job.params)
        if st.button("Відкрити результат", icon=":material/open_in_new:", key="job_open"):
            for key, value in updates.items():
                st.session_state[key] = value
            st.switch_page(page)
    with JobStore(DEFAULT_JOBS_PATH) as js:
        log = js.tail_log(job.id, n=120)
    if log:
        st.code(log, language="text")
    else:
        st.caption("Логу ще немає.")


_jobs_panel()
store.close()
