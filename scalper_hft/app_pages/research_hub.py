"""Research Hub: накопичені результати sweep і walk-forward."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from scalper_hft.app_pages._results import render_sweep_explorer
from scalper_hft.research.sweep_store import SweepStore

st.title("Research Hub")
st.caption(
    "Усі клітинки sweep і walk-forward. Наведіть на колонку — підказка. "
    "**Деталі** відкриває бектест зі свічками, equity і точками входу/виходу."
)

_SWEEP_DB = Path("results") / "sweep.db"

if not _SWEEP_DB.exists():
    st.info("Немає бази результатів (`results/sweep.db`). Спочатку запустіть sweep.")
    st.page_link("app_pages/research.py", label="Дослідження → Sweep matrix", icon=":material/science:")
    st.stop()


@st.cache_data(ttl="10s", show_spinner="Читання sweep.db…")
def load_sweep_results() -> pd.DataFrame:
    try:
        with SweepStore(_SWEEP_DB) as store:
            return store.load()
    except Exception as exc:
        st.error(f"Помилка завантаження результатів sweep: {exc}")
        return pd.DataFrame()


results_df = load_sweep_results()

if results_df.empty:
    st.warning("База результатів порожня.")
    st.page_link("app_pages/research.py", label="Запустити sweep", icon=":material/play_arrow:")
    st.stop()

n_ok = int((results_df["status"] == "ok").sum()) if "status" in results_df.columns else len(results_df)
n_wf = int((results_df["mode"] == "walkforward").sum()) if "mode" in results_df.columns else 0
with st.container(horizontal=True):
    st.metric("Клітинок", len(results_df), border=True)
    st.metric("OK", n_ok, border=True)
    st.metric("Walk-forward", n_wf, border=True)
    if "strategy" in results_df.columns:
        st.metric("Стратегій", int(results_df["strategy"].nunique()), border=True)

render_sweep_explorer(results_df, key_prefix="hub")
