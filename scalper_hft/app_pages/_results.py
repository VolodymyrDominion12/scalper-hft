"""Спільна таблиця результатів sweep/walk-forward для сторінок дашборду."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from scalper_hft.app_pages._common import (
    RESEARCH_CELL_PREFILL,
    RESEARCH_CELL_TAB,
    capacity_job_payload,
    check_strategy_support,
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
    candidate_mask,
    default_sort_column,
    display_columns,
    filter_sweep_results,
    is_audit_action,
    job_matches_combo,
    latest_cells,
    open_action_cells,
)
from scalper_hft.validation.cell_audit import default_train_test

_TABLE_PREFIXES = ("sw_all", "sw_top", "hub", "tq_jobs")


@st.cache_data(ttl=10, show_spinner=False)
def _audit_cell_state() -> dict[tuple[str, str, str], str]:
    """(strategy, symbol, interval) -> стан найсвіжішого аудиту комірки.

    Джерела: черга overfit-задач (queued/running/failed/succeeded) і журнал
    вердиктів (PASS/FAIL — лише для succeeded). 10-секундний TTL: таблиці
    Research Hub / sweep не читають SQLite на кожен rerun.
    """
    state: dict[tuple[str, str, str], str] = {}
    try:
        with JobStore(DEFAULT_JOBS_PATH) as js:
            jobs = js.list_jobs(kind="overfit", limit=1000)
    except Exception:
        jobs = []
    for j in jobs:  # created_at DESC → перший збіг = найсвіжіша задача
        p = j.params or {}
        key = (str(p.get("strategy") or ""), str(p.get("symbol") or ""), str(p.get("interval") or ""))
        if not all(key) or key in state:
            continue
        state[key] = j.status
    if state:
        try:
            from scalper_hft.validation.verdict_store import load_verdicts

            for v in reversed(load_verdicts()):  # найсвіжіший вердикт першим
                key = (str(v.get("strategy") or ""), str(v.get("symbol") or ""), str(v.get("interval") or ""))
                if not all(key) or state.get(key) != "succeeded":
                    continue
                state[key] = "PASS" if v.get("label") == "PASS" else "FAIL"
        except Exception:
            pass
    return state


_AUDIT_LABEL_SHORT = {
    "PASS": "✓ PASS",
    "FAIL": "✗ FAIL",
    "queued": "⏳ queued",
    "running": "▶ running",
    "failed": "✗ failed",
    "cancelled": "—",
}


def audit_cell_label(state: dict[tuple[str, str, str], str], row: Mapping[str, Any]) -> str:
    """Короткий підпис стану аудиту комірки для колонки таблиці."""
    key = (str(row.get("strategy") or ""), str(row.get("symbol") or ""), str(row.get("interval") or ""))
    status = state.get(key)
    if status is None:
        return "—"
    return _AUDIT_LABEL_SHORT.get(status, status)


def enqueue_cell_audits(rows: list[Mapping[str, Any]], *, skip_active: bool = True) -> tuple[int, int]:
    """Поставити overfit-аудит для списку комірок; (запущено, пропущено).

    Той самий fingerprint не дублює queued/running/succeeded; failed
    пере-ставиться автоматично (JobStore.submit).
    """
    state = _audit_cell_state()
    launched = 0
    skipped = 0
    for row in rows:
        combo = combo_prefill(row)
        if not (combo.get("strategy") and combo.get("symbol") and combo.get("interval")):
            continue
        if skip_active and audit_cell_label(state, row) not in {"—", "✗ failed"}:
            skipped += 1
            continue
        train_b, test_b = default_train_test(str(combo["interval"]))
        payload = overfit_job_payload(
            str(combo["strategy"]),
            str(combo["symbol"]),
            str(combo["interval"]),
            int(combo["days"]),
            train_bars=train_b,
            test_bars=test_b,
        )
        job, _alive = submit_research_job("overfit", payload)
        if job.status == "queued":
            launched += 1
        else:
            skipped += 1
    return launched, skipped


def enqueue_cell_backtests(rows: list[Mapping[str, Any]], *, trace: bool = True) -> tuple[int, int]:
    """Поставити повний backtest (за замовчуванням з trace) для вибраних комірок."""
    launched = 0
    skipped = 0
    for row in rows:
        combo = combo_prefill(row)
        if not (combo.get("strategy") and combo.get("symbol") and combo.get("interval")):
            continue
        ok, _ = check_strategy_support(str(combo["strategy"]), is_pair=False)
        if not ok:
            skipped += 1
            continue
        payload = single_backtest_payload(
            str(combo["strategy"]),
            str(combo["symbol"]),
            str(combo["interval"]),
            int(combo["days"]),
            trace=trace,
        )
        job, _alive = submit_research_job("backtest", payload)
        if job.status == "queued":
            launched += 1
        else:
            skipped += 1
    return launched, skipped


def enqueue_cell_capacity(rows: list[Mapping[str, Any]]) -> tuple[int, int]:
    """Поставити capacity для вибраних одиночних комірок."""
    launched = 0
    skipped = 0
    for row in rows:
        combo = combo_prefill(row)
        if not (combo.get("strategy") and combo.get("symbol") and combo.get("interval")):
            continue
        ok, _ = check_strategy_support(str(combo["strategy"]), is_pair=False)
        if not ok:
            skipped += 1
            continue
        payload = capacity_job_payload(
            str(combo["strategy"]),
            str(combo["symbol"]),
            str(combo["interval"]),
            int(combo["days"]),
        )
        job, _alive = submit_research_job("capacity", payload)
        if job.status == "queued":
            launched += 1
        else:
            skipped += 1
    return launched, skipped


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


def open_combo_details(row: Mapping[str, Any], *, enqueue: bool = True, switch: bool = True) -> None:
    """Провалитись на досьє комірки: свічки, equity, точки входу/виходу."""
    combo = combo_prefill(row)
    st.session_state[RESEARCH_CELL_PREFILL] = combo
    st.session_state[RESEARCH_CELL_TAB] = "price"
    if enqueue and combo.get("strategy") and combo.get("symbol") and combo.get("interval"):
        ok, _ = check_strategy_support(str(combo["strategy"]), is_pair=False)
        if ok:
            payload = single_backtest_payload(
                str(combo["strategy"]),
                str(combo["symbol"]),
                str(combo["interval"]),
                int(combo["days"]),
            )
            submit_research_job("backtest", payload)
    if switch:
        st.switch_page("app_pages/cell.py")


def open_combo_audit(row: Mapping[str, Any], *, enqueue: bool = True, switch: bool = True) -> None:
    """Відкрити досьє комірки на вкладці аудиту і поставити overfit job."""
    combo = combo_prefill(row)
    st.session_state[RESEARCH_CELL_PREFILL] = combo
    st.session_state[RESEARCH_CELL_TAB] = "audit"
    if enqueue and combo.get("strategy") and combo.get("symbol") and combo.get("interval"):
        ok, _ = check_strategy_support(str(combo["strategy"]), is_pair=False)
        if ok:
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
    if switch:
        st.switch_page("app_pages/cell.py")


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
    # У Streamlit виклик st.switch_page всередині on_click callback блокується
    # («Calling st.rerun() within a callback is a no-op»).
    # Тому готуємо стан без переходу, а сам switch викликаємо у нормальному циклі рендеру.
    if is_audit_action(label):
        open_combo_audit(row, switch=False)
    else:
        open_combo_details(row, switch=False)
    st.session_state["_results_pending_switch"] = "app_pages/cell.py"


def consume_results_action(key_prefix: str | None = None) -> None:
    """Обробити перехід на досьє комірки / аудит поза межами callback.

    У Streamlit `st.switch_page()` всередині `on_click` callback блокується
    («Calling st.rerun() within a callback is a no-op»). Тому перехід
    виконується в нормальному контексті виконання сторінки.
    """
    target = st.session_state.pop("_results_pending_switch", None)
    if target:
        st.switch_page(target)
        return

    prefixes = [key_prefix] if key_prefix else list(_TABLE_PREFIXES)
    for candidate in prefixes:
        if not candidate:
            continue
        click = st.session_state.get(f"{candidate}_open_btn")
        if click is not None:
            stored = st.session_state.get(f"{candidate}_row_payloads") or []
            row_i = int(getattr(click, "row", -1))
            if 0 <= row_i < len(stored):
                row = stored[row_i]
                label = str(getattr(click, "label", "") or "")
                if candidate == "tq_jobs":
                    st.session_state["tq_override_job_id"] = int(row["id"])
                    return
                if is_audit_action(label):
                    open_combo_audit(row, switch=True)
                else:
                    open_combo_details(row, switch=True)
                return


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
    with_audit: bool = True,
) -> pd.DataFrame:
    """Фільтри + сортування + підказки колонок + клік «Деталі» / «Аудит».

    with_audit=True: додає колонку «Аудит» (стан комірки в черзі/вердикт) і
    дозволяє вибрати рядки → «Запустити аудит N комірок» (batch overfit у чергу).
    """
    if key_prefix not in _TABLE_PREFIXES:
        raise ValueError(f"unknown results table prefix: {key_prefix}")
    consume_results_action(key_prefix)
    st.session_state["_results_active_prefix"] = key_prefix
    if caption:
        st.caption(caption)
    else:
        st.caption(
            "Наведіть на назву колонки — підказка. Клік по заголовку сортує. "
            "**Деталі** відкриває досьє комірки зі свічками. "
            "**Аудит** ставить walk-forward+DSR у чергу."
        )

    source = df
    with st.container(horizontal=True):
        latest_only = st.toggle("Лише останні", value=True, key=f"{key_prefix}_latest")
        cand_only = st.toggle("Кандидати (OOS-гейти)", value=False, key=f"{key_prefix}_cands")
    if latest_only:
        source = latest_cells(source)

    strategies = sorted(source["strategy"].dropna().astype(str).unique()) if "strategy" in source.columns else []
    symbols = sorted(source["symbol"].dropna().astype(str).unique()) if "symbol" in source.columns else []
    intervals = sorted(source["interval"].dropna().astype(str).unique()) if "interval" in source.columns else []
    modes = ["Усі"]
    if "mode" in source.columns:
        modes.extend(sorted(source["mode"].dropna().astype(str).unique()))

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
        source,
        strategies=sel_strats or None,
        symbols=sel_syms or None,
        intervals=sel_ivs or None,
        mode=None if mode == "Усі" else str(mode),
        min_trades=float(min_tr),
        min_sharpe=float(min_sh),
        min_win_rate=float(min_wr),
        query=str(query or ""),
    )
    if cand_only and not view.empty:
        view = view.loc[candidate_mask(view)].reset_index(drop=True)
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
    if with_audit:
        audit_state = _audit_cell_state()
        shown = shown.copy()
        shown["audit"] = [audit_cell_label(audit_state, row) for _, row in view.iterrows()]
    payloads = [row.to_dict() for _, row in view.iterrows()]
    st.session_state[f"{key_prefix}_row_payloads"] = payloads
    column_cfg = sweep_column_config(include_open=True, action_key=f"{key_prefix}_open_btn")
    if with_audit and "audit" in shown.columns:
        column_cfg["audit"] = st.column_config.TextColumn(
            "Аудит",
            help=(
                "Стан аудиту комірки: ✓/✗ PASS/FAIL — останній вердикт; ⏳/▶ queued/running — "
                "задача в черзі; ✗ failed — аудит упав; — аудиту ще не було."
            ),
            width="small",
        )
    if with_audit:
        selection = st.dataframe(
            shown,
            width="stretch",
            hide_index=True,
            column_config=column_cfg,
            key=f"{key_prefix}_grid",
            on_select="rerun",
            selection_mode="multi-row",
        )
        st.caption(f"Показано **{len(shown)}** з {len(source)} рядків.")
        sel_rows = list(selection.selection.rows) if selection is not None and selection.selection else []
        if sel_rows:
            with st.container(border=True):
                st.markdown(f"**Пакетні задачі: вибрано {len(sel_rows)}**")
                _preview = [
                    f"{payloads[int(i)].get('strategy', '?')} · {payloads[int(i)].get('symbol', '?')} · "
                    f"{payloads[int(i)].get('interval', '?')} ({payloads[int(i)].get('days', '?')}d)"
                    for i in sel_rows
                    if 0 <= int(i) < len(payloads)
                ]
                if _preview:
                    shown_preview = "; ".join(_preview[:5]) + (" …" if len(_preview) > 5 else "")
                    st.caption(f"Вибрано: {shown_preview}")
                skip_active = st.toggle(
                    "Пропустити клітинки, що вже мають вердикт або задачу в черзі",
                    value=True,
                    key=f"{key_prefix}_audit_skip",
                )
                rows = [payloads[int(i)] for i in sel_rows if 0 <= int(i) < len(payloads)]
                with st.container(horizontal=True):
                    if st.button(
                        f"Аудит ({len(sel_rows)})",
                        icon=":material/fact_check:",
                        type="primary",
                        key=f"{key_prefix}_audit_btn",
                    ):
                        launched, skipped = enqueue_cell_audits(rows, skip_active=bool(skip_active))
                        st.success(
                            f"Overfit-аудит у черзі: **{launched}**"
                            + (f", пропущено: **{skipped}**" if skipped else "")
                        )
                    if st.button(
                        f"Бектест + trace ({len(sel_rows)})",
                        icon=":material/candlestick_chart:",
                        key=f"{key_prefix}_bt_btn",
                    ):
                        launched, skipped = enqueue_cell_backtests(rows, trace=True)
                        st.success(
                            f"Backtest у черзі: **{launched}**" + (f", пропущено: **{skipped}**" if skipped else "")
                        )
                    if st.button(
                        f"Capacity ({len(sel_rows)})",
                        icon=":material/speed:",
                        key=f"{key_prefix}_cap_btn",
                    ):
                        launched, skipped = enqueue_cell_capacity(rows)
                        st.success(
                            f"Capacity у черзі: **{launched}**" + (f", пропущено: **{skipped}**" if skipped else "")
                        )
                    if st.button(
                        "До порівняння",
                        icon=":material/compare:",
                        key=f"{key_prefix}_cmp_btn",
                    ):
                        bag = list(st.session_state.get("finalist_compare") or [])
                        for row in rows:
                            combo = combo_prefill(row)
                            if combo not in bag:
                                bag.append(combo)
                        st.session_state["finalist_compare"] = bag
                        st.success(f"У порівнянні: **{len(bag)}** комірок.")
                st.page_link("app_pages/jobs.py", label="Черга задач", icon=":material/pending_actions:")
                st.page_link("app_pages/finalists.py", label="Фіналісти", icon=":material/verified:")
    else:
        st.dataframe(
            shown,
            width="stretch",
            hide_index=True,
            column_config=column_cfg,
            key=f"{key_prefix}_grid",
        )
        st.caption(f"Показано **{len(shown)}** з {len(df)} рядків.")
    return view
