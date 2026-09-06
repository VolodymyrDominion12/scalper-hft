"""Індикатор зайнятості дашборду: оверлей на rerun і спінер для важких кроків.

Streamlit вже показує крихітну іконку Running у шапці (~500 мс затримки),
але її легко не помітити. CSS реагує на той самий віджет і робить затемнення
з підписом «Обробка…». Кнопка Stop у шапці лишається клікабельною.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

BUSY_OVERLAY_CSS = """
body:has([data-testid="stStatusWidgetRunningIcon"]) {
  cursor: wait;
}
body:has([data-testid="stStatusWidgetRunningIcon"])::before {
  content: "";
  position: fixed;
  top: 3.75rem;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 999990;
  background: color-mix(in srgb, Canvas 42%, transparent);
  pointer-events: auto;
}
body:has([data-testid="stStatusWidgetRunningIcon"])::after {
  content: "Обробка…";
  position: fixed;
  top: 50%;
  left: 50%;
  z-index: 999991;
  transform: translate(-50%, -50%);
  padding: 0.8rem 1.35rem;
  border: 1px solid color-mix(in srgb, CanvasText 28%, transparent);
  border-radius: 8px;
  background: Canvas;
  color: CanvasText;
  font-weight: 600;
  font-size: 1.05rem;
  letter-spacing: 0.01em;
  pointer-events: none;
}
[data-testid="stStatusWidget"]:has([data-testid="stStatusWidgetRunningIcon"]) {
  outline: 2px solid var(--primary-color, #60a5fa);
  border-radius: 999px;
  padding: 0.1rem 0.35rem;
}
"""


def inject_busy_overlay() -> None:
    """Підключити CSS оверлею на всіх сторінках (викликати з dashboard.py)."""
    import streamlit as st

    st.html(f"<style>\n{BUSY_OVERLAY_CSS}\n</style>")


@contextmanager
def busy(label: str = "Обробка…") -> Iterator[None]:
    """Спінер з часом очікування навколо конкретного важкого кроку."""
    import streamlit as st

    with st.spinner(label, show_time=True):
        yield
