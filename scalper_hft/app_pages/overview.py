"""Сторінка «Моніторинг»: дослідницький бриф — книга стратегій, свіжість даних, paper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from scalper_hft.app_pages._common import SYMBOLS
from scalper_hft.config import get_settings
from scalper_hft.data.cache_ops import (
    action_from_row_label,
    clamp_cache_days,
    delete_symbol_cache,
    download_symbol_cache,
    list_symbol_cache_files,
    max_span_days,
    row_cache_actions,
    suggested_cache_days,
    validate_symbol,
)
from scalper_hft.live.store import PaperStore
from scalper_hft.research.dashboard_brief import (
    cache_inventory,
    compute_paper_kpis,
    inventory_kpis,
    paper_pair_table,
    sweep_highlights,
)
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore
from scalper_hft.research.strategy_book import book_as_rows

settings = get_settings()
data_dir = settings.data_dir_abs
_SWEEP_DB = Path("results") / "sweep.db"
_PAPER_DB = Path("results") / "paper_pairs.sqlite"
_FRESH_BADGE = {
    "fresh": ":green-badge[свіжий]",
    "aging": ":orange-badge[старіє]",
    "stale": ":red-badge[застарілий]",
    "missing": ":gray-badge[немає]",
    "future": ":blue-badge[майбутнє]",
}
_FRESH_FILTERS = {
    "усі": None,
    "застарілі": ("stale", "aging"),
    "немає": ("missing",),
    "свіжі": ("fresh",),
}

st.title("Моніторинг")
st.caption(
    "Дослідницький бриф: чи свіжі дані, що вже валідовано, чи paper підтверджує backtest, і чи тертя не з'їдає edge."
)

with st.container(horizontal=True):
    if settings.dry_run:
        st.badge("Dry-run / paper", icon=":material/science:", color="green")
    else:
        st.badge("LIVE", icon=":material/warning:", color="red")
    st.badge(settings.exchange, icon=":material/account_balance:", color="blue")
    exec_label = "maker" if settings.maker_execution else "taker"
    st.badge(f"Виконання {exec_label}", icon=":material/swap_horiz:", color="violet")
    if settings.data_backend == "postgres":
        st.badge("Кеш PostgreSQL", icon=":material/database:", color="blue")
    else:
        st.badge("Кеш parquet", icon=":material/folder:", color="gray")

st.caption(
    f"Комісії: maker {settings.maker_fee:.2%} · taker {settings.taker_fee:.2%} · "
    f"slippage {settings.slippage_bps:.1f} bps · позиція {settings.position_pct:.1%} · "
    f"пара ≤ {settings.pair_notional_pct:.0%} · портфель пар ≤ {settings.portfolio_notional_pct:.0%}. "
    "Бектест без комісій не є edge."
)

# Multi-exchange KPI + Bot Cards
st.subheader("Системні KPI та Боти")
if _PAPER_DB.exists():
    try:
        with PaperStore(_PAPER_DB) as p_store:
            accounts = p_store.all_accounts()
            bots = p_store.all_bots()

        if not accounts.empty:
            st.caption("Останні баланси по біржах (live/paper)")
            accounts["ts"] = pd.to_datetime(accounts["ts"])
            latest = accounts.sort_values("ts").groupby(["exchange", "mode"]).last().reset_index()

            cols = st.columns(len(latest) if len(latest) > 0 else 1)
            for i, row in latest.iterrows():
                with cols[i % len(cols)]:
                    st.metric(
                        f"{row['exchange']} ({row['mode']})",
                        f"${row['balance']:,.2f}",
                        delta=f"Unrealized PnL: ${row['unrealized_pnl']:.2f}",
                    )

        if not bots.empty:
            st.caption("Активні боти (Bot Cards)")
            bot_cols = st.columns(min(3, len(bots)))
            for i, row in bots.iterrows():
                with bot_cols[i % 3]:
                    st.info(
                        f"**{row['bot_id']}**\n\n"
                        f"Біржа: {row['exchange']} | Режим: {row['mode']}\n\n"
                        f"Останній пінг: {row['last_heartbeat']}"
                    )

    except Exception as e:
        st.warning(f"Не вдалося завантажити Multi-exchange KPI: {e}")

with JobStore(DEFAULT_JOBS_PATH) as _js:
    _worker_alive = _js.worker_is_alive()
if not _worker_alive:
    st.warning("Research worker не запущений — задачі залишаться в черзі.")
    with st.container(horizontal=True):
        st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")
        st.page_link("app_pages/help.py", label="Як запустити worker", icon=":material/help:")
if not _PAPER_DB.exists():
    st.info("Немає paper pairs SQLite. Як зібрати журнал — у Довідці.")
    st.page_link("app_pages/help.py", label="Довідка", icon=":material/menu_book:")


@st.cache_data(ttl="5m", max_entries=8)
def _cached_inventory(data_dir_s: str, symbols: tuple[str, ...], backend: str) -> pd.DataFrame:
    from scalper_hft.data.store import get_store

    store = get_store() if backend == "postgres" else None
    return cache_inventory(Path(data_dir_s), symbols, store=store)


def _clear_cache_pending() -> None:
    st.session_state.cache_pending = None


def _invalidate_inventory() -> None:
    _cached_inventory.clear()
    _clear_cache_pending()


def _safe_symbols(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        try:
            sym = validate_symbol(item)
        except ValueError:
            continue
        if sym not in out:
            out.append(sym)
    return out


def _set_cache_pending(action: str, symbols: list[str]) -> None:
    chosen = _safe_symbols(symbols)
    if not chosen:
        st.toast("Немає коректних символів", icon=":material/warning:")
        return
    st.session_state.cache_pending = {"action": action, "symbols": chosen}


def _on_cache_row_action() -> None:
    click = st.session_state.get("cache_row_action")
    if click is None:
        return
    symbols = st.session_state.get("cache_table_symbols") or []
    row = int(click.row)
    if row < 0 or row >= len(symbols):
        return
    _set_cache_pending(action_from_row_label(str(click.label)), [symbols[row]])


def _render_download_form(symbols: list[str], *, action: str) -> None:
    inv = _cached_inventory(str(data_dir), tuple(SYMBOLS), settings.data_backend)
    span = max_span_days(inv, symbols)
    default_days = suggested_cache_days(span, action=action)
    names = ", ".join(symbols)
    if action == "expand":
        st.caption(
            f"Докачати **префікс** історії для {len(symbols)} симв.: {names}. "
            "Старі бари не стираються; 5m/15m/1h далі ресемпляться з 1m."
        )
    else:
        st.caption(
            f"Докачати **хвіст** 1m для {len(symbols)} симв.: {names}. "
            "Застарілий кеш = lookahead-ризик на «сьогоднішніх» висновках."
        )
    if span is not None and span == span:
        st.caption(f"Поточний span (макс. серед вибраних): {span:.1f} дн")
    days = st.number_input(
        "Глибина, днів",
        min_value=1,
        max_value=1095,
        value=default_days,
        step=30,
        help="Вікно від зараз назад. Оновлення з force докачує хвіст; розширення заповнює дірки на початку.",
    )
    funding = st.toggle("Також funding", value=True)
    trades = st.toggle("Також aggTrades (REST ≈ 2 доби)", value=False)
    force = action != "expand"
    submit = "Розширити" if action == "expand" else "Оновити"
    if st.button(submit, type="primary", icon=":material/download:"):
        try:
            depth = clamp_cache_days(days)
        except ValueError as exc:
            st.error(str(exc))
            return
        errors: list[str] = []
        with st.status(f"Завантаження {len(symbols)} симв.…", expanded=True) as status:
            for i, sym in enumerate(symbols, start=1):
                st.write(f"{i}/{len(symbols)} `{sym}`")
                try:
                    result = download_symbol_cache(sym, depth, force=force, funding=funding, trades=trades)
                    extra = []
                    if result.funding_rows is not None:
                        extra.append(f"funding {result.funding_rows}")
                    if result.trades_rows is not None:
                        extra.append(f"trades {result.trades_rows}")
                    tail = f" · {', '.join(extra)}" if extra else ""
                    st.write(f"  {result.klines_rows} барів 1m{tail}")
                except Exception as exc:  # noqa: BLE001 — мережа/біржа не валить діалог
                    errors.append(f"{sym}: {exc}")
                    st.write(f"  помилка: {exc}")
            if errors:
                status.update(label="Завершено з помилками", state="error")
            else:
                status.update(label="Кеш оновлено", state="complete")
        if errors:
            st.error("\n".join(errors))
            return
        st.toast(f"Оновлено {len(symbols)} симв.", icon=":material/check:")
        _invalidate_inventory()
        st.rerun()


@st.dialog("Оновити кеш", icon=":material/download:", on_dismiss=_clear_cache_pending)
def _refresh_cache_dialog(symbols: list[str]) -> None:
    _render_download_form(symbols, action="refresh")


@st.dialog("Розширити діапазон", icon=":material/calendar_month:", on_dismiss=_clear_cache_pending)
def _expand_cache_dialog(symbols: list[str]) -> None:
    _render_download_form(symbols, action="expand")


@st.dialog("Видалити кеш", icon=":material/delete:", on_dismiss=_clear_cache_pending)
def _delete_cache_dialog(symbols: list[str]) -> None:
    st.warning(f"Видалити parquet-кеш для {len(symbols)} симв.: {', '.join(symbols)}")
    if settings.data_backend == "postgres":
        st.caption(
            "Ринкові дані живуть у PostgreSQL. Це діалог стирає лише файли в data/, "
            "рядки в таблицях klines / funding / agg_trades не чіпає."
        )
    files: list[str] = []
    for sym in symbols:
        files.extend(p.name for p in list_symbol_cache_files(data_dir, sym))
    if files:
        st.caption("Файли, які буде стерто:")
        st.code("\n".join(files), language="text")
    else:
        st.caption("Файлів у data/ немає — нічого видаляти.")
    with st.container(horizontal=True):
        if st.button("Скасувати", icon=":material/close:"):
            _clear_cache_pending()
            st.rerun()
        if st.button("Видалити", type="primary", icon=":material/delete:", disabled=not files):
            deleted: list[str] = []
            for sym in symbols:
                deleted.extend(delete_symbol_cache(data_dir, sym))
            st.toast(f"Видалено файлів: {len(deleted)}", icon=":material/delete:")
            _invalidate_inventory()
            st.rerun()


def _open_cache_dialog() -> None:
    pending = st.session_state.get("cache_pending")
    if not pending:
        return
    symbols = _safe_symbols(list(pending.get("symbols") or []))
    action = str(pending.get("action") or "refresh")
    if not symbols:
        _clear_cache_pending()
        return
    if action == "delete":
        _delete_cache_dialog(symbols)
    elif action == "expand":
        _expand_cache_dialog(symbols)
    else:
        _refresh_cache_dialog(symbols)


@st.fragment
def _cache_manager(inv_df: pd.DataFrame) -> None:
    stale_or_aging: list[str] = []
    missing: list[str] = []
    if not inv_df.empty:
        fresh = inv_df["freshness"].astype(str)
        stale_or_aging = inv_df.loc[fresh.isin(["stale", "aging"]), "symbol"].astype(str).tolist()
        missing = inv_df.loc[fresh.eq("missing"), "symbol"].astype(str).tolist()

    with st.container(horizontal=True, vertical_alignment="bottom"):
        fresh_filter = st.pills(
            "Показати",
            list(_FRESH_FILTERS),
            default="усі",
            required=True,
            key="cache_fresh_filter",
        )
        if st.button(
            "Оновити застарілі",
            icon=":material/sync:",
            disabled=not stale_or_aging,
            help="Докачати хвіст 1m для символів зі свіжістю «старіє» або «застарілий».",
        ):
            _set_cache_pending("refresh", stale_or_aging)
        if st.button(
            "Завантажити відсутні",
            icon=":material/download:",
            disabled=not missing,
            help="Початкове завантаження 1m для символів без кешу.",
        ):
            _set_cache_pending("refresh", missing)

    shown = inv_df.copy()
    wanted = _FRESH_FILTERS.get(fresh_filter or "усі")
    if wanted and not shown.empty:
        shown = shown[shown["freshness"].astype(str).isin(wanted)]
    if shown.empty:
        st.caption("Немає рядків за цим фільтром.")
        _open_cache_dialog()
        return

    shown = shown.reset_index(drop=True)
    shown["freshness"] = shown["freshness"].map(lambda s: _FRESH_BADGE.get(str(s), str(s)))
    shown["actions"] = [row_cache_actions(int(n)) for n in shown["klines_1m"]]
    st.session_state.cache_table_symbols = shown["symbol"].astype(str).tolist()

    event = st.dataframe(
        shown,
        hide_index=True,
        height="content",
        on_select="rerun",
        selection_mode="multi-row",
        key="cache_inv_table",
        column_config={
            "symbol": st.column_config.TextColumn("Символ", pinned=True),
            "klines_1m": st.column_config.NumberColumn("Klines 1m", format="%d"),
            "last_1m": st.column_config.DatetimeColumn("Останній 1m", format="YYYY-MM-DD HH:mm"),
            "span_days": st.column_config.NumberColumn("Span, дні", format="%.1f"),
            "age_hours": None,
            "age_label": st.column_config.TextColumn("Вік"),
            "freshness": st.column_config.MarkdownColumn("Свіжість"),
            "intervals": st.column_config.TextColumn("Інтервали"),
            "n_intervals": st.column_config.NumberColumn("TF", format="%d"),
            "funding": st.column_config.NumberColumn("Funding", format="%d"),
            "funding_last": st.column_config.DatetimeColumn("Funding last", format="YYYY-MM-DD HH:mm"),
            "agg_trades": st.column_config.NumberColumn("aggTrades", format="%d"),
            "book_ticker": st.column_config.NumberColumn("Book ticker", format="%d"),
            "actions": st.column_config.ButtonColumn(
                "Дії",
                help="Оновити хвіст, розширити історію або видалити parquet-файли символу.",
                on_click=_on_cache_row_action,
                key="cache_row_action",
            ),
        },
    )
    selection = getattr(event, "selection", None)
    selected_idx = list(getattr(selection, "rows", []) or [])
    selected = [str(shown.iloc[i]["symbol"]) for i in selected_idx if 0 <= i < len(shown)]

    if selected:
        st.caption(f"Вибрано: {', '.join(selected)}")
        with st.container(horizontal=True):
            if st.button("Оновити", icon=":material/download:", key="cache_bulk_refresh"):
                _set_cache_pending("refresh", selected)
            if st.button("Розширити діапазон", icon=":material/calendar_month:", key="cache_bulk_expand"):
                _set_cache_pending("expand", selected)
            if st.button("Видалити", icon=":material/delete:", key="cache_bulk_delete"):
                _set_cache_pending("delete", selected)
    else:
        st.caption("Позначте рядки для масових дій або відкрийте меню в колонці «Дії».")

    _open_cache_dialog()


@st.cache_data(ttl="30s", max_entries=4)
def _cached_sweep(path_s: str) -> pd.DataFrame:
    from scalper_hft.research.sweep_store import SweepStore

    path = Path(path_s)
    if not path.exists():
        return pd.DataFrame()
    try:
        with SweepStore(path) as store:
            return store.load()
    except Exception:  # noqa: BLE001 — битий sweep.db не валить сторінку
        return pd.DataFrame()


inv_slot = st.container()
with inv_slot.skeleton():
    inv = _cached_inventory(str(data_dir), tuple(SYMBOLS), settings.data_backend)
kpis = inventory_kpis(inv)

with st.container(horizontal=True):
    st.metric("Символів у всесвіті", kpis["symbols"], border=True, help="Канонічний список дашборду")
    st.metric(
        "З 1m кешем",
        kpis["with_1m"],
        delta=f"немає: {kpis['missing']}" if kpis["missing"] else None,
        delta_color="inverse" if kpis["missing"] else "off",
        border=True,
    )
    st.metric(
        "Застарілих 1m",
        kpis["stale"],
        delta=">24 год без нового бара" if kpis["stale"] else "ок",
        delta_color="inverse" if kpis["stale"] else "off",
        border=True,
        help="Свіжий < 2 год, старіючий < 24 год, далі — застарілий",
    )
    st.metric("Барів 1m", f"{kpis['klines_1m']:,}", border=True)

if kpis["stale"] or kpis["missing"]:
    st.info(
        "Кеш 1m застарілий або відсутній — оновіть таблицю нижче. "
        "Застарілі бари = lookahead-ризик на «сьогоднішніх» висновках."
    )
    st.page_link("app_pages/help.py", label="Як оновити кеш", icon=":material/help:")

st.header(":material/menu_book: Книга стратегій")
st.caption("Консолідований статус після walk-forward / DSR / paper. Не плутати з одним in-sample бектестом.")
book_df = pd.DataFrame(book_as_rows())
lane_filter = st.pills(
    "Смуга",
    ["усі", "валідована", "дослідження", "відхилена", "чекає даних"],
    default="усі",
    required=True,
    key="book_lane",
)
shown = book_df if lane_filter == "усі" else book_df[book_df["lane"] == lane_filter]
st.dataframe(
    shown.drop(columns=["lane_key"]),
    hide_index=True,
    column_config={
        "strategy": st.column_config.TextColumn("Стратегія", pinned=True),
        "pair": st.column_config.TextColumn("Пара"),
        "lane": st.column_config.TextColumn("Статус"),
        "headline": st.column_config.TextColumn("Висновок"),
        "conditions": st.column_config.TextColumn("Умови"),
    },
)

st.header(":material/database: Свіжість кешу")
if settings.data_backend == "postgres":
    st.caption(
        "Джерело: PostgreSQL (`DATA_BACKEND=postgres`), таблиці klines / funding / agg_trades — "
        "не файли `data/*.parquet`. Span і вік останнього 1m бара. "
        "Застарілий кеш = lookahead-ризик на «сьогоднішніх» висновках."
    )
else:
    st.caption(
        "Span і вік останнього 1m бара з parquet у data/. Позначте рядки для масових дій або меню «Дії» в рядку. "
        "Застарілий кеш = lookahead-ризик на «сьогоднішніх» висновках."
    )
st.session_state.setdefault("cache_pending", None)
_cache_manager(inv)

st.header(":material/monitoring: Paper pairs")
if _PAPER_DB.exists():
    with PaperStore(_PAPER_DB) as store:
        stats = store.fill_stats()
        eq = store.all_equity()
        months = store.all_months()
        recent_orders = store.recent_orders()

    k = compute_paper_kpis(stats, eq)
    fill_delta = None if k.fill_rate != k.fill_rate else f"{k.fill_rate:.0%} fill-rate"
    spark = k.sparkline if len(k.sparkline) >= 2 else None
    with st.container(horizontal=True):
        st.metric("Filled", k.filled, border=True)
        st.metric(
            "Unfilled",
            k.unfilled,
            delta=fill_delta,
            delta_color="off",
            border=True,
            help="Fill-rate = filled / (filled + unfilled). Низький fill-rate з'їдає paper-edge pairs_arb.",
        )
        st.metric("Pending", k.pending, border=True)
        last_eq = sum(k.last_equity_by_pair.values()) if k.last_equity_by_pair else None
        st.metric(
            "Equity (сума пар)",
            f"{last_eq:,.0f}" if last_eq is not None else "—",
            border=True,
            chart_data=spark,
            chart_type="area",
            help="Сума останніх точок по парах, не єдиний рахунок",
        )

    if k.last_ts is not None:
        st.caption(f"Останній запис equity: {k.last_ts:%Y-%m-%d %H:%M} UTC · точок {k.n_equity_points}")

    pairs_tbl = paper_pair_table(eq)
    if not pairs_tbl.empty:
        st.dataframe(
            pairs_tbl,
            hide_index=True,
            column_config={
                "pair": st.column_config.TextColumn("Пара", pinned=True),
                "start": st.column_config.NumberColumn("Старт", format="%.2f"),
                "last": st.column_config.NumberColumn("Остання equity", format="%.2f"),
                "return": st.column_config.NumberColumn("Дохідність", format="percent"),
                "spark": st.column_config.LineChartColumn("Equity"),
                "n_points": st.column_config.NumberColumn("Точок", format="%d"),
            },
        )
        if "ts" in eq.columns and "equity" in eq.columns and "pair" in eq.columns:
            eq_plot = eq.copy()
            eq_plot["ts"] = pd.to_datetime(eq_plot["ts"])
            pivot_eq = eq_plot.pivot_table(index="ts", columns="pair", values="equity", aggfunc="last")
            st.line_chart(pivot_eq, height=280, x_label="Час", y_label="Equity")

    if months is not None and not months.empty:
        with st.container(border=True):
            st.markdown("**Місячний PnL**")
            st.caption("Стоп на пару після двох збиткових місяців поспіль (правило розгортання pairs_arb).")
            month_pivot = months.pivot_table(index="pair", columns="month", values="pnl", aggfunc="sum")
            st.dataframe(
                month_pivot,
                column_config={
                    col: st.column_config.NumberColumn(str(col), format="%.2f") for col in month_pivot.columns
                },
            )

    if not recent_orders.empty:
        with st.expander("Останні ордери", icon=":material/list:"):
            st.dataframe(recent_orders.head(40), hide_index=True)
else:
    st.caption("Немає `results/paper_pairs.sqlite` — запустіть: `uv run python -m scalper_hft.cli paper-replay-pairs`")

st.header(":material/grid_on: Останній sweep")
sweep_df = _cached_sweep(str(_SWEEP_DB))
if sweep_df.empty:
    st.caption(
        "Немає `results/sweep.db`. Запуск — сторінка «Дослідження» або `uv run python -m scalper_hft.cli sweep`."
    )
else:
    n_ok = int((sweep_df["status"] == "ok").sum()) if "status" in sweep_df.columns else len(sweep_df)
    st.caption(f"Клітинок у store: {len(sweep_df)} · успішних: {n_ok}. Сортування за OOS Sharpe, не in-sample.")
    top = sweep_highlights(sweep_df, top_n=8)
    if top.empty:
        st.caption("Немає рядків з Sharpe для підсвітки.")
    else:
        st.dataframe(
            top,
            hide_index=True,
            column_config={
                "strategy": st.column_config.TextColumn("Стратегія", pinned=True),
                "n_trades": st.column_config.NumberColumn("Угод", format="%d"),
                "sharpe": st.column_config.NumberColumn("Sharpe IS", format="%.3f"),
                "avg_oos_sharpe": st.column_config.NumberColumn("OOS Sharpe", format="%.3f"),
                "oos_positive_frac": st.column_config.NumberColumn("OOS+ частка", format="percent"),
                "max_dd": st.column_config.NumberColumn("Max DD", format="percent"),
                "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
            },
        )

st.header(":material/show_chart: Paper-run CSV")
results_dir = Path("results")
if results_dir.exists():
    eq_files = sorted(results_dir.glob("paper_equity_*.csv"))
    if eq_files:
        combined_eq: dict[str, pd.Series] = {}
        for eq_file in eq_files:
            csv_eq = pd.read_csv(eq_file, parse_dates=["ts"]).set_index("ts")
            combined_eq[eq_file.stem.replace("paper_equity_", "")] = csv_eq["equity"]
        st.line_chart(pd.DataFrame(combined_eq), height=280, x_label="Час", y_label="Equity")
    else:
        st.caption("Немає CSV paper-run.")
else:
    st.caption("Немає директорії `results/`")
