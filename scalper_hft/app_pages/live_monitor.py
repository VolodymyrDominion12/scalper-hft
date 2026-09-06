"""Сторінка «Live / WebSocket»: субсекундний моніторинг та ручне керування ботами."""

from __future__ import annotations

import logging

import httpx
import streamlit as st

from scalper_hft.api.auth import create_access_token
from scalper_hft.config import get_settings
from scalper_hft.live.control import load_control, save_control
from scalper_hft.live.store import PaperStore
from scalper_hft.visualization.ws_component import ws_live_monitor_component

logger = logging.getLogger(__name__)
settings = get_settings()

st.title("Live / WebSocket моніторинг")
st.caption(
    "Реал-тайм моніторинг відкритих позицій, перевірка затримки (ping RTT) "
    "та ручне керування ботами через FastAPI WebSocket і REST API."
)

# ─── 1. Верхній статусний рядок ───────────────────────────────────────────────
with st.container(horizontal=True):
    if settings.dry_run:
        st.badge("Dry-run / paper", icon=":material/science:", color="green")
    else:
        st.badge("LIVE TRADING", icon=":material/warning:", color="red")
    st.badge(settings.exchange.upper(), icon=":material/account_balance:", color="blue")
    exec_label = "maker" if settings.maker_execution else "taker"
    st.badge(f"Виконання {exec_label}", icon=":material/swap_horiz:", color="violet")

    ctrl_state = load_control()
    if ctrl_state.flatten:
        st.badge("FLATTEN ACTIVE", icon=":material/emergency_home:", color="red")
    elif ctrl_state.pause:
        st.badge("ТОРГІВЛЯ НА ПАУЗІ", icon=":material/pause_circle:", color="orange")
    elif ctrl_state.no_new_entries:
        st.badge("НОВІ ВХОДИ ЗАБЛОКОВАНО", icon=":material/block:", color="yellow")
    else:
        st.badge("ТОРГІВЛЯ АКТИВНА", icon=":material/check_circle:", color="green")


# ─── 2. Налаштування API та токена ───────────────────────────────────────────
if "live_api_base_url" not in st.session_state:
    st.session_state.live_api_base_url = "http://localhost:8080"
if "live_ws_url" not in st.session_state:
    st.session_state.live_ws_url = "ws://localhost:8080/ws"

# Авто-генерація валідного JWT токена
if "live_jwt_token" not in st.session_state or not st.session_state.live_jwt_token:
    try:
        st.session_state.live_jwt_token = create_access_token(
            {"sub": "dashboard_admin", "role": "admin"},
            expires_delta_hours=24,
        )
    except Exception:
        st.session_state.live_jwt_token = ""

with st.expander("⚙️ Параметри підключення до API та WebSocket", expanded=False):
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.session_state.live_api_base_url = st.text_input(
            "REST API URL",
            value=st.session_state.live_api_base_url,
            help="Базовий URL FastAPI бекенду",
        )
    with col_u2:
        st.session_state.live_ws_url = st.text_input(
            "WebSocket URL",
            value=st.session_state.live_ws_url,
            help="WebSocket адреса для потокових оновлень",
        )

    st.session_state.live_jwt_token = st.text_input(
        "JWT Access Token",
        value=st.session_state.live_jwt_token,
        type="password",
        help="Токен авторизації (автоматично згенеровано з API_SECRET_KEY)",
    )

    btn_test = st.button("Перевірити REST зв'язок з сервером", icon=":material/wifi_find:")
    if btn_test:
        headers = {"Authorization": f"Bearer {st.session_state.live_jwt_token}"}
        try:
            resp = httpx.get(
                f"{st.session_state.live_api_base_url}/api/v1/status",
                headers=headers,
                timeout=3.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success(
                    f"✅ Зв'язок успішний! Статус: {data.get('status')}. "
                    f"Активних WS клієнтів: {data.get('active_ws_connections', 0)}"
                )
            else:
                st.error(f"❌ Сервер повернув помилку {resp.status_code}: {resp.text}")
        except Exception as exc:
            st.error(f"❌ Не вдалося підключитися до {st.session_state.live_api_base_url}: {exc}")


# ─── 3. KPI показники (Auto-refresh кожні 5 секунд) ───────────────────────────
@st.fragment(run_every="5s")
def render_kpis() -> None:
    store = PaperStore()
    positions = store.open_positions()
    pos_count = len(positions) if hasattr(positions, "__len__") else 0

    total_pnl = 0.0
    if hasattr(positions, "iterrows"):
        for _, p in positions.iterrows():
            total_pnl += float(p.get("unrealized_pnl", 0.0) or 0.0)

    acc = store.latest_account(exchange=settings.exchange, mode="paper" if settings.dry_run else "live")
    balance = acc["balance"] if acc else 0.0
    margin_used = acc["margin_used"] if acc else 0.0

    bots_df = store.all_bots()
    active_bots = len(bots_df[bots_df["status"] == "running"]) if hasattr(bots_df, "empty") and not bots_df.empty else 0

    with st.container(horizontal=True):
        pnl_delta = f"{'+' if total_pnl >= 0 else ''}${total_pnl:.2f}"
        st.metric("Нереалізований PnL", f"${total_pnl:,.2f}", delta=pnl_delta, border=True)
        st.metric("Відкритих позицій", str(pos_count), border=True)
        st.metric("Баланс рахунку", f"${balance:,.2f}", border=True)
        st.metric("Використана маржа", f"${margin_used:,.2f}", border=True)
        st.metric("Активних ботів", str(active_bots), border=True)


render_kpis()


# ─── 4. Субсекундний WebSocket Монітор (CCv2) ─────────────────────────────────
st.subheader("Реал-тайм WebSocket потік")

ws_live_monitor_component(
    ws_url=st.session_state.live_ws_url,
    token=st.session_state.live_jwt_token,
    key="ws_live_monitor_widget",
)


# ─── 5. Панель ручного керування та Kill-Switch ───────────────────────────────
with st.container(border=True):
    st.subheader("🎮 Панель ручного керування ботами та позиціями")
    st.caption("Керування режимом торгівлі демона через `control.json` та аварійне закриття позицій.")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("⏸️ Призупинити (Pause)", width="stretch", help="Призупинити всі операції бота"):
            try:
                save_control(pause=True)
                st.toast("Торгівлю призупинено (pause=true)", icon=":material/pause:")
                st.rerun()
            except Exception as e:
                st.error(f"Помилка: {e}")

    with col_btn2:
        if st.button("▶️ Відновити (Resume)", width="stretch", help="Відновити звичайну торгівлю"):
            try:
                save_control(pause=False, no_new_entries=False, flatten=False)
                st.toast("Торгівлю відновлено!", icon=":material/play_arrow:")
                st.rerun()
            except Exception as e:
                st.error(f"Помилка: {e}")

    with col_btn3:
        if st.button("🚫 Заборонити нові входи", width="stretch", help="Дозволити тільки закриття відкритих позицій"):
            try:
                save_control(no_new_entries=True)
                st.toast("Нові входи заборонено (no_new_entries=true)", icon=":material/block:")
                st.rerun()
            except Exception as e:
                st.error(f"Помилка: {e}")

    st.divider()

    # Аварійний Kill-Switch
    col_ks1, col_ks2 = st.columns([2, 1])
    with col_ks1:
        st.markdown("**🚨 Аварійний Kill-Switch (Emergency Flatten)**")
        st.caption("Миттєво активує режим `flatten`, зупиняє нові входи і закриває всі відкриті позиції в сховищі.")
    with col_ks2:
        confirm_flatten = st.checkbox("Підтверджую екстрене закриття", key="confirm_flatten")
        if st.button("Закрити всі позиції (Flatten)", type="primary", disabled=not confirm_flatten, width="stretch"):
            headers = {"Authorization": f"Bearer {st.session_state.live_jwt_token}"}
            try:
                # Викликаємо REST ендпоінт бекенду
                resp = httpx.post(
                    f"{st.session_state.live_api_base_url}/api/v1/emergency/flatten",
                    headers=headers,
                    timeout=4.0,
                )
                if resp.status_code == 200:
                    st.success("✅ Всі позиції закрито, торгівлю зупинено.")
                else:
                    # Локальний fallback якщо бекенд не запущено
                    save_control(pause=True, flatten=True)
                    st.warning("Бекенд недоступний, оновлено локальний control.json (flatten=True).")
                st.rerun()
            except Exception as exc:
                save_control(pause=True, flatten=True)
                st.warning(f"REST API помилка ({exc}), активовано локальний control.json.")
                st.rerun()

    st.divider()

    # Закриття конкретної позиції
    st.markdown("**Закриття окремої позиції**")
    store = PaperStore()
    pos_df = store.open_positions()
    open_symbols = list(pos_df["symbol"].unique()) if hasattr(pos_df, "empty") and not pos_df.empty else []

    col_cp1, col_cp2 = st.columns([2, 1])
    with col_cp1:
        selected_symbol = st.selectbox(
            "Оберіть позицію для закриття",
            options=open_symbols if open_symbols else ["Немає відкритих позицій"],
            disabled=len(open_symbols) == 0,
        )
    with col_cp2:
        if st.button("Закрити вибрану", width="stretch", disabled=len(open_symbols) == 0):
            headers = {"Authorization": f"Bearer {st.session_state.live_jwt_token}"}
            try:
                resp = httpx.post(
                    f"{st.session_state.live_api_base_url}/api/v1/positions/close",
                    headers=headers,
                    json={"symbol": selected_symbol, "exchange": settings.exchange, "mode": "paper"},
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    st.success(f"✅ Позицію {selected_symbol} закрито.")
                else:
                    store.log_position(exchange=settings.exchange, symbol=selected_symbol, side="flat", size=0.0, entry_price=0.0, mark_price=0.0, unrealized_pnl=0.0, mode="paper")
                    st.info(f"Оновлено локально: позицію {selected_symbol} закрито.")
                st.rerun()
            except Exception:
                store.log_position(exchange=settings.exchange, symbol=selected_symbol, side="flat", size=0.0, entry_price=0.0, mark_price=0.0, unrealized_pnl=0.0, mode="paper")
                st.info(f"Закрито в локальному сховищі: {selected_symbol}.")
                st.rerun()


# ─── 6. Симуляція сигналів (Paper Demo) ───────────────────────────────────────
with st.expander("🧪 Симуляція тестових сигналів (Швидка перевірка WebSocket)", expanded=False):
    st.caption("Натисніть кнопку, щоб створити тестову позицію і спостерігати миттєве оновлення віджета без перезавантаження сторінки.")
    col_d1, col_d2, col_d3 = st.columns(3)

    headers = {"Authorization": f"Bearer {st.session_state.live_jwt_token}"}

    with col_d1:
        if st.button("➕ Симулювати LONG BTCUSDT", width="stretch"):
            try:
                httpx.post(
                    f"{st.session_state.live_api_base_url}/api/v1/paper/mock_position",
                    headers=headers,
                    json={
                        "symbol": "BTCUSDT",
                        "side": "long",
                        "size": 0.25,
                        "entry_price": 64500.0,
                        "unrealized_pnl": 42.50,
                    },
                    timeout=3.0,
                )
                st.toast("Створено LONG BTCUSDT!", icon=":material/trending_up:")
            except Exception:
                # Прямий запис у базу, якщо REST офлайн
                store = PaperStore()
                store.log_position(exchange="binance", symbol="BTCUSDT", side="long", size=0.25, entry_price=64500.0, mark_price=64700.0, unrealized_pnl=42.50, mode="paper")
                st.toast("Записано в локальний PaperStore!", icon=":material/database:")

    with col_d2:
        if st.button("➕ Симулювати SHORT ETHUSDT", width="stretch"):
            try:
                httpx.post(
                    f"{st.session_state.live_api_base_url}/api/v1/paper/mock_position",
                    headers=headers,
                    json={
                        "symbol": "ETHUSDT",
                        "side": "short",
                        "size": 2.5,
                        "entry_price": 3450.0,
                        "unrealized_pnl": -15.80,
                    },
                    timeout=3.0,
                )
                st.toast("Створено SHORT ETHUSDT!", icon=":material/trending_down:")
            except Exception:
                store = PaperStore()
                store.log_position(exchange="binance", symbol="ETHUSDT", side="short", size=2.5, entry_price=3450.0, mark_price=3470.0, unrealized_pnl=-15.80, mode="paper")
                st.toast("Записано в локальний PaperStore!", icon=":material/database:")

    with col_d3:
        if st.button("➕ Симулювати LONG XRPUSDT", width="stretch"):
            try:
                httpx.post(
                    f"{st.session_state.live_api_base_url}/api/v1/paper/mock_position",
                    headers=headers,
                    json={
                        "symbol": "XRPUSDT",
                        "side": "long",
                        "size": 1500.0,
                        "entry_price": 0.58,
                        "unrealized_pnl": 18.20,
                    },
                    timeout=3.0,
                )
                st.toast("Створено LONG XRPUSDT!", icon=":material/trending_up:")
            except Exception:
                store = PaperStore()
                store.log_position(exchange="binance", symbol="XRPUSDT", side="long", size=1500.0, entry_price=0.58, mark_price=0.60, unrealized_pnl=18.20, mode="paper")
                st.toast("Записано в локальний PaperStore!", icon=":material/database:")


# ─── 7. REST Телеметрія та Діагностика ────────────────────────────────────────
st.subheader("Деталізована телеметрія (REST API)")
tab_acc, tab_bots, tab_trades, tab_sys = st.tabs(["Рахунки", "Боти", "Останні угоди", "Системний статус"])

with tab_acc:
    store = PaperStore()
    acc_df = store.recent_accounts(limit=10)
    if hasattr(acc_df, "empty") and not acc_df.empty:
        st.dataframe(acc_df, width="stretch", hide_index=True)
    else:
        st.info("Немає записів балансів рахунків. Вони формуються під час роботи paper/live демона.")

with tab_bots:
    bots_df = store.all_bots()
    if hasattr(bots_df, "empty") and not bots_df.empty:
        st.dataframe(bots_df, width="stretch", hide_index=True)
    else:
        st.info("Немає зареєстрованих ботів у сховищі.")

with tab_trades:
    trades_df = store.all_trades()
    if hasattr(trades_df, "empty") and not trades_df.empty:
        st.dataframe(trades_df.tail(20).iloc[::-1], width="stretch", hide_index=True)
    else:
        st.info("Немає угод у журналі.")

with tab_sys:
    st.json({
        "exchange": settings.exchange,
        "dry_run": settings.dry_run,
        "maker_execution": settings.maker_execution,
        "maker_fee": settings.maker_fee,
        "taker_fee": settings.taker_fee,
        "slippage_bps": settings.slippage_bps,
        "api_secret_configured": bool(settings.api_secret_key),
        "control_file": str(load_control()),
    })
