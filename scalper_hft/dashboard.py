"""Головний вхід Streamlit-дашборду scalper-hft (мультисторінка).

Запуск (завжди через venv проєкту, не Anaconda `streamlit` з PATH):

    uv pip install -e ".[dashboard]"
    uv run python -m scalper_hft.cli dashboard

Сторінки (app_pages/):
    - «Моніторинг» — кеш даних по символах та paper-результати;
    - «Бектест» — запуск бектесту, інтерактивний графік угод (свічки +
      індикатори + точки входу/виходу + SL/TP, клік по маркеру → деталі
      угоди) та діагностика (cohort / stress / capacity);
    - «Дослідження» — масовий sweep, filter attribution, порівняння
      equity кривих, MAE/MFE аналіз, hourly heatmap, топ комбінації.
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

import streamlit as st  # noqa: E402

st.set_page_config(page_title="scalper-hft", page_icon="📈", layout="wide")

pages = [
    st.Page("app_pages/overview.py", title="Моніторинг", icon=":material/monitoring:", default=True),
    st.Page("app_pages/backtest.py", title="Бектест", icon=":material/query_stats:"),
    st.Page("app_pages/research.py", title="Дослідження", icon=":material/science:"),
]
st.navigation(pages).run()
