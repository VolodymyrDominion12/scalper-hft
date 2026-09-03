"""Сторінка «Моніторинг»: дослідницький бриф — книга стратегій, свіжість даних, paper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from scalper_hft.app_pages._common import SYMBOLS
from scalper_hft.config import get_settings
from scalper_hft.live.store import PaperStore
from scalper_hft.research.dashboard_brief import (
    cache_inventory,
    compute_paper_kpis,
    inventory_kpis,
    paper_pair_table,
    sweep_highlights,
)
from scalper_hft.research.strategy_book import book_as_rows

settings = get_settings()
data_dir = settings.data_dir_abs
_SWEEP_DB = Path("results") / "sweep.db"
_PAPER_DB = Path("results") / "paper_pairs.sqlite"

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

st.caption(
    f"Комісії: maker {settings.maker_fee:.2%} · taker {settings.taker_fee:.2%} · "
    f"slippage {settings.slippage_bps:.1f} bps · позиція {settings.position_pct:.1%} · "
    f"пара ≤ {settings.pair_notional_pct:.0%} · портфель пар ≤ {settings.portfolio_notional_pct:.0%}. "
    "Бектест без комісій не є edge."
)


@st.cache_data(ttl="5m", max_entries=8)
def _cached_inventory(data_dir_s: str, symbols: tuple[str, ...]) -> pd.DataFrame:
    return cache_inventory(Path(data_dir_s), symbols)


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
    inv = _cached_inventory(str(data_dir), tuple(SYMBOLS))
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
st.caption("Span і вік останнього 1m бара. Застарілий кеш = lookahead-ризик на «сьогоднішніх» висновках.")
fresh_map = {"fresh": "свіжий", "aging": "старіє", "stale": "застарілий", "missing": "немає", "future": "майбутнє"}
inv_view = inv.copy()
if not inv_view.empty:
    inv_view["freshness"] = inv_view["freshness"].map(lambda s: fresh_map.get(str(s), str(s)))
st.dataframe(
    inv_view,
    hide_index=True,
    column_config={
        "symbol": st.column_config.TextColumn("Символ", pinned=True),
        "klines_1m": st.column_config.NumberColumn("Klines 1m", format="%d"),
        "last_1m": st.column_config.DatetimeColumn("Останній 1m", format="YYYY-MM-DD HH:mm"),
        "span_days": st.column_config.NumberColumn("Span, дні", format="%.1f"),
        "age_hours": None,
        "age_label": st.column_config.TextColumn("Вік"),
        "freshness": st.column_config.TextColumn("Свіжість"),
        "intervals": st.column_config.TextColumn("Інтервали"),
        "n_intervals": st.column_config.NumberColumn("TF", format="%d"),
        "funding": st.column_config.NumberColumn("Funding", format="%d"),
        "funding_last": st.column_config.DatetimeColumn("Funding last", format="YYYY-MM-DD HH:mm"),
        "agg_trades": st.column_config.NumberColumn("aggTrades", format="%d"),
        "book_ticker": st.column_config.NumberColumn("Book ticker", format="%d"),
    },
)

st.header(":material/monitoring: Paper pairs")
if _PAPER_DB.exists():
    store = PaperStore(_PAPER_DB)
    try:
        stats = store.fill_stats()
        eq = store.all_equity()
        months = store.all_months()
        recent_orders = store.recent_orders()
    finally:
        store.close()

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
