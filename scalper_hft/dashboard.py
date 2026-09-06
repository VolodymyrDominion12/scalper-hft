"""Головний вхід Streamlit-дашборду scalper-hft (мультисторінка).

Запуск (завжди через venv проєкту, не Anaconda `streamlit` з PATH):

    uv pip install -e ".[dashboard]"
    uv run python -m scalper_hft.cli dashboard

Сторінки (app_pages/):
    - «Моніторинг» — кеш даних по символах та paper-результати;
    - «Бектест» — постановка бектесту в чергу, інтерактивний графік угод;
    - «Дослідження» — масовий sweep (enqueue), filter attribution, порівняння;
    - «Задачі» — черга job, логи, cancel / rerun;
    - «Довідка» — посібник користувача.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit додає каталог скрипта (scalper_hft/), а не корінь репо.
# Без цього `from scalper_hft...` у app_pages падає ModuleNotFoundError,
# якщо пакет не встановлено в той самий інтерпретатор (часто Anaconda PATH).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scalper_hft.app_pages._busy  # noqa: E402, F401
import scalper_hft.app_pages._common  # noqa: E402, F401
from scalper_hft.app_pages import reload_shared  # noqa: E402

reload_shared()

import streamlit as st  # noqa: E402

from scalper_hft.app_pages._busy import inject_busy_overlay  # noqa: E402
from scalper_hft.dashboard_auth import check_password  # noqa: E402

st.set_page_config(page_title="scalper-hft", page_icon="📈", layout="wide")

if not check_password():
    st.stop()

inject_busy_overlay()

pages = [
    st.Page("app_pages/overview.py", title="Моніторинг", icon=":material/monitoring:", default=True),
    st.Page("app_pages/live_monitor.py", title="Live / WebSocket", icon=":material/sensors:"),
    st.Page("app_pages/multi_exchange.py", title="Multi-Exchange", icon=":material/stacked_line_chart:"),
    st.Page("app_pages/backtest.py", title="Бектест", icon=":material/query_stats:"),
    st.Page("app_pages/research.py", title="Дослідження", icon=":material/science:"),
    st.Page("app_pages/research_hub.py", title="Research Hub", icon=":material/library_books:"),
    st.Page("app_pages/jobs.py", title="Задачі", icon=":material/pending_actions:"),
    st.Page("app_pages/help.py", title="Довідка", icon=":material/help:"),
]
st.navigation(pages).run()
