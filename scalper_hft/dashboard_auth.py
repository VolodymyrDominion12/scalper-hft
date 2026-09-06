"""Автентифікація для Streamlit Dashboard.

Використовує bcrypt для перевірки хешу паролю (DASHBOARD_PASSWORD_HASH).
Якщо пароль не налаштовано або порожній, автентифікація пропускається.
"""

from __future__ import annotations

import logging
import os
import secrets

import bcrypt
import streamlit as st

from scalper_hft.config import get_settings

logger = logging.getLogger(__name__)

# Секретний ключ для підпису cookie/токенів (якщо буде потрібно в майбутньому).
# Поки що достатньо session_state, оскільки Streamlit зберігає його на бекенді.
_SESSION_TOKEN_KEY = "auth_session_token"


def stored_password_hash() -> str:
    """Хеш пароля дашборду: Settings, інакше змінна середовища.

    Streamlit може тримати старий синглтон `Settings` без поля
    `dashboard_password_hash`. Тоді читаємо `DASHBOARD_PASSWORD_HASH` напряму,
    щоб не падати з AttributeError і не відкривати дашборд, якщо хеш уже в .env.
    """
    settings = get_settings()
    raw = getattr(settings, "dashboard_password_hash", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return (os.getenv("DASHBOARD_PASSWORD_HASH") or "").strip()


def check_password() -> bool:
    """Повертає `True`, якщо користувач успішно ввійшов.

    Якщо DASHBOARD_PASSWORD_HASH не налаштовано (порожній), повертає True одразу.
    """
    stored_hash = stored_password_hash()

    # Якщо хеш не задано — вільний доступ
    if not stored_hash:
        return True

    # Перевірка наявності токену сесії
    if st.session_state.get(_SESSION_TOKEN_KEY):
        return True

    # Форма логіну
    with st.form("Логін", clear_on_submit=True):
        st.write("🔒 **Доступ обмежено**")
        st.write("Введіть пароль для доступу до дашборду.")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Увійти")

        if submitted:
            if not password:
                st.error("Будь ласка, введіть пароль.")
                return False

            try:
                # Перевіряємо хеш
                is_valid = bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
                if is_valid:
                    # Успішний логін
                    st.session_state[_SESSION_TOKEN_KEY] = secrets.token_hex(16)
                    st.rerun()
                else:
                    st.error("❌ Невірний пароль")
            except ValueError as exc:
                logger.error("Помилка перевірки bcrypt хешу: %s", exc)
                st.error("Помилка конфігурації сервера (невірний формат хешу).")

    return False


def generate_hash(password: str) -> str:
    """Генерує bcrypt хеш для заданого паролю (для CLI утиліти)."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
