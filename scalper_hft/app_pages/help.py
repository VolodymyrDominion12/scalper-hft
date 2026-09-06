"""Сторінка «Довідка»: посібник користувача дашборду."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from scalper_hft.config import get_settings
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore

_DOC = Path(__file__).resolve().parents[2] / "docs" / "DASHBOARD.md"
_SWEEP_DB = Path("results") / "sweep.db"

settings = get_settings()
st.title("Довідка")
st.caption("Як користуватись дашбордом. Рахунок іде в worker, не в браузері.")

with JobStore(DEFAULT_JOBS_PATH) as js:
    alive = js.worker_is_alive()

with st.container(horizontal=True):
    if settings.dry_run:
        st.badge("Dry-run / paper", icon=":material/science:", color="green")
    else:
        st.badge("LIVE", icon=":material/warning:", color="red")
    if alive:
        st.badge("воркер активний", icon=":material/check_circle:", color="green")
    else:
        st.badge("воркер не запущений", icon=":material/pause_circle:", color="orange")
    st.badge(settings.exchange, icon=":material/account_balance:", color="blue")

st.caption(f"jobs: `{DEFAULT_JOBS_PATH}` · sweep: `{_SWEEP_DB}` · кеш: `{settings.data_dir_abs}`")
if not alive:
    st.warning("Запустіть `uv run python -m scalper_hft.cli job worker` або дашборд без `--no-worker`.")

if _DOC.exists():
    text = _DOC.read_text(encoding="utf-8")
    if text.startswith("# "):
        text = text.split("\n", 1)[1]
    st.markdown(text)
else:
    st.error(f"Немає файлу `{_DOC}`.")
