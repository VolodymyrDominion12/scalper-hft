"""Спільна таблиця результатів sweep/walk-forward для сторінок дашборду."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from scalper_hft.app_pages._common import (
    RESEARCH_AUDIT_PREFILL,
    RESEARCH_BT_PREFILL,
    RESEARCH_SECTION,
    combo_prefill,
    overfit_job_payload,
    single_backtest_payload,
    submit_research_job,
)
from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, Job, JobStore
from scalper_hft.research.results_table import (
    SWEEP_COLUMN_BY_KEY,
    TRADE_COLUMNS,
    ColumnHint,
    default_sort_column,
    display_columns,
    filter_sweep_results,
    is_audit_action,
    job_matches_combo,
    open_action_cells,
)
from scalper_hft.validation.cell_audit import default_train_test

_TABLE_PREFIXES = ("sw_all", "sw_top", "hub", "tq_jobs")


def _hint_config(hint: ColumnHint) -> object:
    if hint.kind == "text":
        return st.column_config.TextColumn(hint.label, help=hint.help, pinned=hint.pinned or None)
    if hint.kind == "datetime":
        return st.column_config.DatetimeColumn(
            hint.label, help=hint.help, format=hint.number_format or "YYYY-MM-DD HH:mm"
        )
    if hint.kind == "int":
        return st.column_config.NumberColumn(
            hint.label, help=hint.help, format=hint.number_format or "%d", pinned=hint.pinned or None
        )
    if hint.kind == "percent":
        return st.column_config.NumberColumn(
            hint.label, help=hint.help, format=hint.number_format or "percent", pinned=hint.pinned or None
        )
    return st.column_config.NumberColumn(
        hint.label, help=hint.help, format=hint.number_format or "%.3f", pinned=hint.pinned or None
    )


def sweep_column_config(*, include_open: bool = True, action_key: str = "sw_all_open_btn") -> dict[str, object]:
    """column_config з підказками для кожної метрики sweep."""
    cfg: dict[str, object] = {key: _hint_config(hint) for key, hint in SWEEP_COLUMN_BY_KEY.items()}
    cfg["id"] = None
    cfg["filter_attribution"] = None
    if include_open:
        cfg["open"] = st.column_config.ButtonColumn(
            "Відкрити",
            help="Деталі — свічки, equity, входи/виходи. Аудит — walk-forward + DSR (лише для walkforward).",
            on_click=_on_results_row_action,
            key=action_key,
            pinned=True,
        )
    return cfg


def trade_column_config() -> dict[str, object]:
    """Підказки колонок таблиці угод на сторінці бектесту."""
    return {hint.key: _hint_config(hint) for hint in TRADE_COLUMNS}


def open_combo_details(row: Mapping[str, Any], *, enqueue: bool = True) -> None:
    """Провалитись на Бектест: свічки, equity, точки входу/виходу.

    Sweep не зберігає угоди — тому за потреби ставимо повний backtest job.
    """
    combo = combo_prefill(row)
    st.session_state[RESEARCH_BT_PREFILL] = combo
    if enqueue and combo.get("strategy") and combo.get("symbol") and combo.get("interval"):
        payload = single_backtest_payload(
            str(combo["strategy"]),
            str(combo["symbol"]),
            str(combo["interval"]),
            int(combo["days"]),
        )
        submit_research_job("backtest", payload)
    st.switch_page("app_pages/backtest.py")


def open_combo_audit(row: Mapping[str, Any], *, enqueue: bool = True) -> None:
    """Відкрити аудит комірки (walk-forward + DSR) і поставити overfit job."""
    combo = combo_prefill(row)
    st.session_state[RESEARCH_AUDIT_PREFILL] = combo
    st.session_state[RESEARCH_SECTION] = "Аудит комірки"
    if enqueue and combo.get("strategy") and combo.get("symbol") and combo.get("interval"):
        train_b, test_b = default_train_test(str(combo["interval"]))
        payload = overfit_job_payload(
            str(combo["strategy"]),
            str(combo["symbol"]),
            str(combo["interval"]),
            int(combo["days"]),
            train_bars=train_b,
            test_bars=test_b,
        )
        submit_research_job("overfit", payload)
    st.switch_page("app_pages/research.py")


def _on_results_row_action() -> None:
    click = None
    prefix = ""
    for candidate in _TABLE_PREFIXES:
        value = st.session_state.get(f"{candidate}_open_btn")
        if value is not None:
            click = value
            prefix = candidate
            break
    if click is None:
        return
    stored = st.session_state.get(f"{prefix}_row_payloads") or []
    row_i = int(getattr(click, "row", -1))
    if row_i < 0 or row_i >= len(stored):
        return
    row = stored[row_i]
    label = str(getattr(click, "label", "") or "")
    if prefix == "tq_jobs":
        st.session_state["tq_override_job_id"] = int(row["id"])
        return
    if is_audit_action(label):
        open_combo_audit(row)
        return
    open_combo_details(row)


def list_combo_jobs(
    *,
    strategy: str,
    symbol: str,
    interval: str,
    days: int | None = None,
    limit: int = 200,
) -> list[Job]:
    """Знайти backtest-задачі для комбінації (будь-який статус)."""
    with JobStore(DEFAULT_JOBS_PATH) as js:
        jobs = js.list_jobs(kind="backtest", limit=limit)
    return [
        j for j in jobs if job_matches_combo(j.params, strategy=strategy, symbol=symbol, interval=interval, days=days)
    ]


_SHOW_QUALITY_LABEL = ":material/analytics: Показати"


def render_backtest_job_picker(jobs: list[Job], *, key_prefix: str = "tq_jobs") -> None:
    """Таблиця наявних бектестів: «Якість угод» не залежить лише від слайдера днів."""
    if not jobs:
        return
    table = pd.DataFrame(
        [
            {
                "id": j.id,
                "days": j.params.get("days"),
                "status": j.status,
                "created": (j.created_at or "")[:19],
                "error": (j.error or "")[:80],
            }
            for j in jobs
        ]
    )
    table["open"] = _SHOW_QUALITY_LABEL
    st.session_state[f"{key_prefix}_row_payloads"] = table[["id"]].to_dict("records")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("Job", format="%d", help="Ідентифікатор задачі в черзі."),
            "days": st.column_config.NumberColumn("Днів", format="%d", help="Глибина історії цього бектесту."),
            "status": st.column_config.TextColumn("Статус", help="succeeded — можна відкрити аналіз угод."),
            "created": st.column_config.TextColumn("Створено", help="Час постановки задачі (UTC)."),
            "error": st.column_config.TextColumn("Помилка", help="Текст, якщо задача failed."),
            "open": st.column_config.ButtonColumn(
                "Якість",
                help="Показати MAE/MFE і розподіл PnL цього бектесту на цій сторінці.",
                on_click=_on_results_row_action,
                key=f"{key_prefix}_open_btn",
            ),
        },
        key=f"{key_prefix}_grid",
    )


def render_sweep_explorer(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    caption: str | None = None,
    min_trades: int | None = None,
    min_sharpe: float | None = None,
    min_win_rate: float | None = None,
    default_mode: str = "Усі",
) -> pd.DataFrame:
    """Фільтри + сортування + підказки колонок + клік «Деталі» / «Аудит»."""
    if key_prefix not in _TABLE_PREFIXES:
        raise ValueError(f"unknown results table prefix: {key_prefix}")
    st.session_state["_results_active_prefix"] = key_prefix
    if caption:
        st.caption(caption)
    else:
        st.caption(
            "Наведіть на назву колонки — підказка. Клік по заголовку сортує. "
            "**Деталі** відкриває бектест зі свічками й угодами (sweep зберігає лише метрики)."
        )

    strategies = sorted(df["strategy"].dropna().astype(str).unique()) if "strategy" in df.columns else []
    symbols = sorted(df["symbol"].dropna().astype(str).unique()) if "symbol" in df.columns else []
    intervals = sorted(df["interval"].dropna().astype(str).unique()) if "interval" in df.columns else []
    modes = ["Усі"]
    if "mode" in df.columns:
        modes.extend(sorted(df["mode"].dropna().astype(str).unique()))

    with st.container(horizontal=True, vertical_alignment="bottom"):
        query = st.text_input(
            "Пошук",
            key=f"{key_prefix}_q",
            placeholder="стратегія, символ, ТФ",
        )
        sel_strats = st.multiselect("Стратегія", strategies, key=f"{key_prefix}_strats")
        sel_syms = st.multiselect("Символ", symbols, key=f"{key_prefix}_syms")
        sel_ivs = st.multiselect("Таймфрейм", intervals, key=f"{key_prefix}_ivs")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        mode_idx = modes.index(default_mode) if default_mode in modes else 0
        mode = st.selectbox("Режим", modes, index=mode_idx, key=f"{key_prefix}_mode")
        trades_default = 0 if min_trades is None else int(min_trades)
        sharpe_default = -10.0 if min_sharpe is None else float(min_sharpe)
        wr_default = 0.0 if min_win_rate is None else float(min_win_rate)
        min_tr = st.number_input("Min угод", 0, 100_000, trades_default, key=f"{key_prefix}_mintr")
        min_sh = st.number_input("Min Sharpe", -10.0, 20.0, sharpe_default, step=0.1, key=f"{key_prefix}_minsh")
        min_wr = st.number_input("Min win rate", 0.0, 1.0, wr_default, step=0.05, key=f"{key_prefix}_minwr")

    view = filter_sweep_results(
        df,
        strategies=sel_strats or None,
        symbols=sel_syms or None,
        intervals=sel_ivs or None,
        mode=None if mode == "Усі" else str(mode),
        min_trades=float(min_tr),
        min_sharpe=float(min_sh),
        min_win_rate=float(min_wr),
        query=str(query or ""),
    )
    if view.empty:
        st.warning("Немає рядків після фільтрів.")
        return view

    sort_candidates = [
        c for c in (default_sort_column(view), "sharpe", "n_trades", "total_return") if c in view.columns
    ]
    sort_options = list(dict.fromkeys([*sort_candidates, *[c for c in display_columns(view) if c in view.columns]]))
    with st.container(horizontal=True, vertical_alignment="bottom"):
        sort_col = st.selectbox("Сортувати за", sort_options, key=f"{key_prefix}_sort")
        sort_asc = st.toggle("За зростанням", value=False, key=f"{key_prefix}_asc")
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=bool(sort_asc), na_position="last").reset_index(drop=True)

    cols = display_columns(view)
    shown = view[cols].copy()
    modes = shown["mode"].astype(str).tolist() if "mode" in shown.columns else [None] * len(shown)
    shown["open"] = open_action_cells(modes)
    payloads = [row.to_dict() for _, row in view.iterrows()]
    st.session_state[f"{key_prefix}_row_payloads"] = payloads
    st.dataframe(
        shown,
        width="stretch",
        hide_index=True,
        column_config=sweep_column_config(include_open=True, action_key=f"{key_prefix}_open_btn"),
        key=f"{key_prefix}_grid",
    )
    st.caption(f"Показано **{len(shown)}** з {len(df)} рядків.")
    return view
