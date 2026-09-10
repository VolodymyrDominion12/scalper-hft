"""Спільний графік угод/спреду з робочим вікном (не весь період)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from scalper_hft.app_pages._busy import busy
from scalper_hft.app_pages._results import trade_column_config
from scalper_hft.features.indicators import add_standard_features
from scalper_hft.visualization import (
    auto_indicator_columns,
    default_price_window,
    find_trade_by_ts,
    make_backtest_figure,
    make_pairs_figure,
    neighboring_entry_ts,
    shift_window,
    trade_detail_figure,
    trades_table,
    window_around_trade,
)

_VIEW_BARS = 250


def _py(ts: object) -> object:
    return pd.Timestamp(ts).to_pydatetime()


def _ensure_window(
    *,
    key_prefix: str,
    index: pd.DatetimeIndex,
    trades: pd.DataFrame | None,
    src: str,
) -> None:
    src_key = f"{key_prefix}_window_src"
    win_key = f"{key_prefix}_window"
    pending = st.session_state.pop(f"{key_prefix}_pending_window", None)
    if pending is not None:
        st.session_state[win_key] = pending
        st.session_state[src_key] = src
        return
    if st.session_state.get(src_key) == src and win_key in st.session_state:
        return
    start, end = default_price_window(index, trades, bars=_VIEW_BARS)
    st.session_state[win_key] = (_py(start), _py(end))
    st.session_state[src_key] = src


def _apply_window(key_prefix: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Нове вікно застосовується на початку наступного прогону, до слайдера."""
    st.session_state[f"{key_prefix}_pending_window"] = (_py(start), _py(end))


def _pan(key_prefix: str, index: pd.DatetimeIndex, direction: int) -> None:
    cur = st.session_state.get(f"{key_prefix}_window")
    if not cur:
        return
    start, end = shift_window(cur[0], cur[1], index, direction)
    _apply_window(key_prefix, start, end)


def _jump_trade(key_prefix: str, index: pd.DatetimeIndex, entry_ts: object) -> None:
    start, end = window_around_trade(pd.Timestamp(entry_ts), index, bars=_VIEW_BARS)
    _apply_window(key_prefix, start, end)
    st.session_state[f"{key_prefix}_sel_ts"] = pd.Timestamp(entry_ts)


def _step_trade(key_prefix: str, index: pd.DatetimeIndex, trades: pd.DataFrame, step: int) -> None:
    current = st.session_state.get(f"{key_prefix}_sel_ts")
    nxt = neighboring_entry_ts(trades, current, step=step)
    if nxt is None:
        return
    _jump_trade(key_prefix, index, nxt)


@st.fragment
def render_backtest_chart(view: dict[str, Any], *, key_prefix: str = "bt") -> None:
    """Графік угод: робоче вікно, пан, стрибок до угоди."""
    df, res, title = view["df"], view["res"], view["title"]
    src = f"{title}:{len(df)}:{df.index[0]}:{df.index[-1]}"
    _ensure_window(key_prefix=key_prefix, index=df.index, trades=res.trades, src=src)

    st.subheader("Графік угод")
    left, right = st.columns([1, 4])
    with left:
        ts0 = df.index[0].to_pydatetime()
        ts1 = df.index[-1].to_pydatetime()
        st.caption("Вікно графіка (~250 барів від першого входу)")
        window = st.slider(
            "Час",
            min_value=ts0,
            max_value=ts1,
            format="%d.%m %H:%M",
            key=f"{key_prefix}_window",
        )
        with st.container(horizontal=True):
            st.button(
                "Назад",
                icon=":material/chevron_left:",
                key=f"{key_prefix}_pan_left",
                on_click=_pan,
                args=(key_prefix, df.index, -1),
            )
            st.button(
                "Далі",
                icon=":material/chevron_right:",
                key=f"{key_prefix}_pan_right",
                on_click=_pan,
                args=(key_prefix, df.index, 1),
            )
        with st.container(horizontal=True):
            st.button(
                "Попер. угода",
                icon=":material/skip_previous:",
                key=f"{key_prefix}_prev_tr",
                on_click=_step_trade,
                args=(key_prefix, df.index, res.trades, -1),
                disabled=res.trades is None or res.trades.empty,
            )
            st.button(
                "Наст. угода",
                icon=":material/skip_next:",
                key=f"{key_prefix}_next_tr",
                on_click=_step_trade,
                args=(key_prefix, df.index, res.trades, 1),
                disabled=res.trades is None or res.trades.empty,
            )
        with_trades = st.toggle("Точки входу/виходу", value=True, key=f"{key_prefix}_trades")
        with_sl_tp = st.toggle("Рівні SL / TP", value=True, key=f"{key_prefix}_sl_tp")
        with_inds = st.toggle("Індикатори", value=True, key=f"{key_prefix}_inds")
        max_bars = st.select_slider(
            "Максимум барів",
            options=[1_000, 5_000, 20_000, 100_000, 500_000],
            value=20_000,
            key=f"{key_prefix}_max_bars",
        )
        n_tr = 0 if res.trades is None else len(res.trades)
        st.caption(f"Угод: {n_tr} · Max DD: {res.metrics.max_drawdown:.1%}")
        st.caption("Клік по маркеру або рядку таблиці центрує вікно на угоді.")
    with right:
        with busy("Будую графік угод…"):
            fdf = add_standard_features(df)
            fig = make_backtest_figure(
                fdf,
                res,
                symbol=title,
                start=window[0],
                end=window[1],
                max_bars=max_bars,
                with_trades=with_trades,
                with_sl_tp=with_sl_tp,
                indicators=auto_indicator_columns(fdf) if with_inds else [],
            )
        sel = st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_fig", on_select="rerun", selection_mode="points")
        sel_state: Any = getattr(sel, "selection", None)
        if sel_state:
            points = sel_state.get("points") if isinstance(sel_state, dict) else getattr(sel_state, "points", None)
            for p in points or []:
                if not (isinstance(p, dict) and p.get("x") is not None):
                    continue
                hit = find_trade_by_ts(res.trades, p["x"])
                if hit is None:
                    continue
                entry = hit["entry_ts"]
                if st.session_state.get(f"{key_prefix}_sel_ts") != pd.Timestamp(entry):
                    _jump_trade(key_prefix, df.index, entry)
                    st.rerun(scope="fragment")
                break

    sel_ts = st.session_state.get(f"{key_prefix}_sel_ts")
    trade = find_trade_by_ts(res.trades, sel_ts) if sel_ts is not None else None
    if trade is not None:
        t0 = pd.Timestamp(trade["entry_ts"])
        with st.container(border=True):
            st.subheader(f"Деталі угоди · {t0:%d.%m.%Y %H:%M}")
            sl = trade["sl_price"] if "sl_price" in trade.index else float("nan")
            tp = trade["tp_price"] if "tp_price" in trade.index else float("nan")
            with st.container(horizontal=True):
                st.metric("Сторона", "Лонг" if trade["side"] == 1 else "Шорт", border=True)
                st.metric("Вхід → Вихід", f"{trade['entry_price']:.2f} → {trade['exit_price']:.2f}", border=True)
                st.metric("PnL", f"{trade['ret']:.3%}", border=True)
                st.metric("SL / TP", f"{sl:.2f} / {tp:.2f}" if pd.notna(sl) and pd.notna(tp) else "—", border=True)
            st.plotly_chart(
                trade_detail_figure(fdf, res, trade["entry_ts"]),
                width="stretch",
                key=f"{key_prefix}_detail_fig",
            )
            if st.button("Закрити деталі", icon=":material/close:", key=f"{key_prefix}_clear_sel"):
                st.session_state[f"{key_prefix}_sel_ts"] = None
                st.rerun(scope="fragment")

    st.subheader("Угоди")
    if res.trades is not None and not res.trades.empty:
        table = trades_table(res, initial_capital=10_000.0)
        picked = st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config=trade_column_config(),
            key=f"{key_prefix}_trades_grid",
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = list(picked.selection.rows) if picked is not None and picked.selection else []
        if rows:
            entry = table.iloc[int(rows[0])]["Вхід"]
            if pd.notna(entry) and st.session_state.get(f"{key_prefix}_sel_ts") != pd.Timestamp(entry):
                _jump_trade(key_prefix, df.index, entry)
                st.rerun(scope="fragment")
    else:
        st.info("Угод за цей період немає — спробуйте іншу стратегію/період.")


@st.fragment
def render_pairs_chart(res: Any, title: str, *, key_prefix: str = "pairs") -> None:
    """Спред + робоче вікно для pairs_arb."""
    if res.spread is None or getattr(res.spread, "empty", True):
        st.warning("Немає спреду в артефактах.")
        return
    src = f"{title}:{len(res.spread)}:{res.spread.index[0]}:{res.spread.index[-1]}"
    _ensure_window(key_prefix=key_prefix, index=res.spread.index, trades=res.trades, src=src)

    st.subheader("Графік спреду")
    left, right = st.columns([1, 4])
    with left:
        ts0 = res.spread.index[0].to_pydatetime()
        ts1 = res.spread.index[-1].to_pydatetime()
        st.caption("Вікно графіка (~250 барів від першого входу)")
        window = st.slider(
            "Час",
            min_value=ts0,
            max_value=ts1,
            format="%d.%m %H:%M",
            key=f"{key_prefix}_window",
        )
        with st.container(horizontal=True):
            st.button(
                "Назад",
                icon=":material/chevron_left:",
                key=f"{key_prefix}_pan_left",
                on_click=_pan,
                args=(key_prefix, res.spread.index, -1),
            )
            st.button(
                "Далі",
                icon=":material/chevron_right:",
                key=f"{key_prefix}_pan_right",
                on_click=_pan,
                args=(key_prefix, res.spread.index, 1),
            )
        with st.container(horizontal=True):
            st.button(
                "Попер. угода",
                icon=":material/skip_previous:",
                key=f"{key_prefix}_prev_tr",
                on_click=_step_trade,
                args=(key_prefix, res.spread.index, res.trades, -1),
                disabled=res.trades is None or res.trades.empty,
            )
            st.button(
                "Наст. угода",
                icon=":material/skip_next:",
                key=f"{key_prefix}_next_tr",
                on_click=_step_trade,
                args=(key_prefix, res.spread.index, res.trades, 1),
                disabled=res.trades is None or res.trades.empty,
            )
        with_trades = st.toggle("Точки входу/виходу", value=True, key=f"{key_prefix}_trades")
        max_bars = st.select_slider(
            "Максимум барів",
            options=[1_000, 5_000, 20_000, 100_000, 500_000],
            value=20_000,
            key=f"{key_prefix}_max_bars",
        )
        n_tr = 0 if res.trades is None else len(res.trades)
        st.caption(f"Угод: {n_tr} · Max DD: {res.metrics.max_drawdown:.1%}")
    with right:
        with busy("Будую графік спреду…"):
            fig = make_pairs_figure(
                res,
                start=window[0],
                end=window[1],
                max_bars=max_bars,
                with_trades=with_trades,
                symbol=title,
            )
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_fig")
    st.subheader("Угоди")
    if res.trades is not None and not res.trades.empty:
        table = trades_table(res, initial_capital=10_000.0)
        picked = st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config=trade_column_config(),
            key=f"{key_prefix}_trades_grid",
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = list(picked.selection.rows) if picked is not None and picked.selection else []
        if rows:
            entry = table.iloc[int(rows[0])]["Вхід"]
            if pd.notna(entry) and st.session_state.get(f"{key_prefix}_sel_ts") != pd.Timestamp(entry):
                _jump_trade(key_prefix, res.spread.index, entry)
                st.rerun(scope="fragment")
    else:
        st.info("Угод за цей період немає.")
