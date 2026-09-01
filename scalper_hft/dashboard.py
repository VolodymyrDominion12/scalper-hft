"""Головний вхід Streamlit-дашборду scalper-hft (мультисторінка).

Запуск:
    uv pip install -e ".[dashboard]"
    .venv/bin/streamlit run scalper_hft/dashboard.py

Сторінки (app_pages/):
    - «Моніторинг» — кеш даних по символах та paper-результати;
    - «Бектест» — запуск бектесту, інтерактивний графік угод (свічки +
      індикатори + точки входу/виходу + SL/TP, клік по маркеру → деталі
      угоди) та діагностика (cohort / stress / capacity).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="scalper-hft", page_icon="📈", layout="wide")

pages = [
    st.Page("app_pages/overview.py", title="Моніторинг", icon=":material/monitoring:", default=True),
    st.Page("app_pages/backtest.py", title="Бектест", icon=":material/query_stats:"),
]
st.navigation(pages).run()
