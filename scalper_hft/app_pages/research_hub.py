"""Research Hub: Історія бектестів, результати sweep-прогонів та звітів."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Research Hub")
st.caption("Накопичені результати бектестів, walk-forward аналізу та параметричних sweep-ів.")

_SWEEP_DB = Path("results") / "sweep.db"

if not _SWEEP_DB.exists():
    st.info("Немає бази результатів (results/sweep.db). Запустіть бектести або sweep.")
    st.stop()

@st.cache_data(ttl="10s")
def load_sweep_results() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(str(_SWEEP_DB))
        df = pd.read_sql_query("SELECT * FROM sweep_results ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()
        return df
    except Exception as exc:
        st.error(f"Помилка завантаження результатів sweep: {exc}")
        return pd.DataFrame()

results_df = load_sweep_results()

if results_df.empty:
    st.warning("База результатів порожня.")
    st.stop()

st.subheader("Останні бектести")
st.dataframe(results_df, use_container_width=True, hide_index=True)
