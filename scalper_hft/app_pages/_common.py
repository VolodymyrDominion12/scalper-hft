"""Спільні константи сторінок дашборду.

SYMBOLS — канонічний універсум інструментів (див. `scalper_hft.symbols`),
той самий, що є fallback-списком `DEFAULT_SYMBOLS` у config. Якщо у .env
задано свій `DEFAULT_SYMBOLS`, CLI/sweep/MCP підуть за ним, а дашборд
показує курований список нижче. PAIR_CHOICES — окремий курований список
(лише перевірені пари для pairs_arb).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal

from scalper_hft.symbols import CANONICAL_SYMBOLS

if TYPE_CHECKING:
    from scalper_hft.research.jobs import Job

SYMBOLS: list[str] = list(CANONICAL_SYMBOLS)

# Пари для pairs_arb: лише ті, що пройшли коінтеграційний скринінг/валідацію.
PAIR_CHOICES = ["XRPUSDT/BTCUSDT", "BTCUSDT/ETHUSDT", "LINKUSDT/BTCUSDT", "LINKUSDT/ETHUSDT"]

# Ті самі ТФ, що validation.sweep.DEFAULT_INTERVALS — без імпорту validation
# (дашборд піднімає _common до site-packages).
BT_INTERVALS: list[str] = ["1m", "5m", "15m", "30m", "1h", "4h"]
RESEARCH_BT_PREFILL = "research_bt_prefill"
RESEARCH_AUDIT_PREFILL = "research_audit_prefill"
RESEARCH_CELL_PREFILL = "research_cell_prefill"
RESEARCH_CELL_TAB = "research_cell_tab"
RESEARCH_SECTION = "research_section_prefill"
RESEARCH_SECTIONS: tuple[str, ...] = (
    "Масовий пошук",
    "Повний цикл",
    "Аудит комірки",
)
JOB_STATUSES: tuple[str, ...] = ("queued", "running", "succeeded", "failed", "cancelled")
JOB_KINDS: tuple[str, ...] = ("backtest", "pairs", "sweep", "overfit", "capacity")

JobLevel = Literal["empty", "info", "warning", "error", "ok"]


def combo_prefill(row: Mapping[str, Any], *, default_days: int = 60) -> dict[str, Any]:
    """Нормалізувати рядок sweep/топа до prefill для бектесту й аудиту."""
    days_raw = row.get("days", default_days)
    try:
        days_f = float(days_raw)  # type: ignore[arg-type]
        days = default_days if days_f != days_f else int(days_f)
    except (TypeError, ValueError):
        days = default_days
    out: dict[str, Any] = {
        "strategy": str(row.get("strategy") or ""),
        "symbol": str(row.get("symbol") or ""),
        "interval": str(row.get("interval") or ""),
        "days": days,
    }
    if row.get("leg1") and row.get("leg2"):
        out["pair"] = f"{row['leg1']}/{row['leg2']}"
    elif row.get("pair"):
        out["pair"] = str(row["pair"])
    return out


def _apply_days(state: MutableMapping[str, Any], raw: Mapping[str, Any], key: str) -> None:
    if raw.get("days") is None:
        return
    try:
        state[key] = int(raw["days"])
    except (TypeError, ValueError):
        pass


def apply_research_bt_prefill(state: MutableMapping[str, Any]) -> None:
    """Перенести prefill зі сторінки досліджень у ключі віджетів бектесту.

    Викликати до створення sidebar-віджетів.
    """
    raw = state.pop(RESEARCH_BT_PREFILL, None)
    if not isinstance(raw, dict):
        return
    if raw.get("strategy"):
        state["bt_strategy"] = str(raw["strategy"])
    if raw.get("symbol"):
        state["bt_symbol"] = str(raw["symbol"])
    pair = raw.get("pair")
    if not pair and raw.get("leg1") and raw.get("leg2"):
        pair = f"{raw['leg1']}/{raw['leg2']}"
    if pair:
        state["bt_pair"] = str(pair)
    if raw.get("interval"):
        iv = str(raw["interval"])
        state["bt_interval_single"] = iv
        state["bt_interval_pairs"] = iv
    _apply_days(state, raw, "bt_days")


def apply_shared_research_keys(state: MutableMapping[str, Any], raw: Mapping[str, Any]) -> None:
    """Записати спільні rs_* ключі (стратегія / символ / ТФ / днів)."""
    if raw.get("strategy"):
        state["rs_strategy"] = str(raw["strategy"])
    if raw.get("symbol"):
        state["rs_symbol"] = str(raw["symbol"])
    if raw.get("interval"):
        state["rs_interval"] = str(raw["interval"])
    _apply_days(state, raw, "rs_days")


def apply_research_audit_prefill(state: MutableMapping[str, Any]) -> None:
    """Перенести prefill топа у спільний рядок досліджень. Викликати до віджетів."""
    raw = state.pop(RESEARCH_AUDIT_PREFILL, None)
    if not isinstance(raw, dict):
        return
    apply_shared_research_keys(state, raw)


def apply_research_section_prefill(state: MutableMapping[str, Any]) -> None:
    """Відкрити потрібний розділ досліджень (після switch_page з «Задач»)."""
    raw = state.pop(RESEARCH_SECTION, None)
    if isinstance(raw, str) and raw in RESEARCH_SECTIONS:
        state["research_section"] = raw


def check_strategy_support(strategy_name: str, *, is_pair: bool = False) -> tuple[bool, str | None]:
    """Перевіряє, чи підтримує стратегія запуск у поточному режимі UI (одиночний символ або пара).

    Returns:
        (True, None) якщо конфігурація валідна.
        (False, error_message) якщо стратегія вимагає іншого типу даних.
    """
    if not strategy_name:
        return True, None
    try:
        from scalper_hft.strategies import get_strategy

        strat = get_strategy(strategy_name)
    except Exception:
        return True, None

    req = getattr(strat, "requires", frozenset())
    if "basket" in req:
        return (
            False,
            f"Стратегія '{strategy_name}' потребує кошика активів (basket_df ≥2 колонок) і не підтримується для одиночного бектесту.",
        )

    if "multi_symbol" in req:
        if is_pair and strategy_name == "pairs_arb":
            return True, None
        if strategy_name == "pairs_arb":
            return (
                False,
                "Стратегія 'pairs_arb' потребує дві ноги (leg1/leg2). Використовуйте сторінку «Комірка» у режимі пари.",
            )
        if strategy_name == "cross_momentum":
            return (
                False,
                "Стратегія 'cross_momentum' є крос-секційною (потребує зрізу цін ≥3 інструментів або кошика) "
                "і не підтримується для одиночного бектесту.",
            )
        return False, f"Стратегія '{strategy_name}' вимагає мульти-символьних даних ({sorted(req)})."

    if is_pair and strategy_name != "pairs_arb":
        return False, f"Стратегія '{strategy_name}' розрахована на одиночний символ, а не на пару."

    return True, None


def overfit_job_payload(
    strategy: str,
    symbol: str,
    interval: str,
    days: int,
    *,
    train_bars: int,
    test_bars: int,
    validate: bool = True,
) -> dict[str, Any]:
    if validate:
        ok, err = check_strategy_support(strategy, is_pair=False)
        if not ok:
            raise ValueError(err)
    return {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "days": int(days),
        "train_bars": int(train_bars),
        "test_bars": int(test_bars),
    }


def single_backtest_payload(
    strategy: str,
    symbol: str,
    interval: str,
    days: int,
    *,
    trace: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Той самий payload, що сторінка Бектест для одиночної стратегії."""
    if validate:
        ok, err = check_strategy_support(strategy, is_pair=False)
        if not ok:
            raise ValueError(err)
    return {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "days": int(days),
        "params": {},
        "maker": False,
        "trace": bool(trace),
        "base_interval": "1m",
        "breakeven_gate": False,
    }


def capacity_job_payload(
    strategy: str, symbol: str, interval: str, days: int, *, validate: bool = True
) -> dict[str, Any]:
    if validate:
        ok, err = check_strategy_support(strategy, is_pair=False)
        if not ok:
            raise ValueError(err)
    return {
        "strategy": strategy,
        "symbol": symbol,
        "interval": interval,
        "days": int(days),
        "params": {},
        "maker": False,
        "base_interval": "1m",
        "scales": [1.0, 2.0, 5.0, 10.0],
    }


def job_label(kind: str, params: Mapping[str, Any] | None) -> str:
    """Короткий підпис задачі для таблиці черги."""
    p = dict(params or {})
    if kind == "pairs":
        return f"pairs · {p.get('leg1', '?')}/{p.get('leg2', '?')} {p.get('interval', '')}".strip()
    if kind == "sweep":
        n_s = len(p.get("strategies") or [])
        n_y = len(p.get("symbols") or [])
        n_i = len(p.get("intervals") or [])
        return f"sweep · {n_s}×{n_y}×{n_i} {p.get('mode', '')}".strip()
    strat = str(p.get("strategy") or kind)
    symbol = str(p.get("symbol") or "")
    interval = str(p.get("interval") or "")
    return " · ".join(part for part in (strat, f"{symbol} {interval}".strip()) if part)


def combo_from_job_params(kind: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Prefill для бектесту/аудиту з params збереженої job."""
    row: dict[str, Any] = dict(params)
    row.setdefault("strategy", "pairs_arb" if kind == "pairs" else params.get("strategy", ""))
    return combo_prefill(row)


def apply_research_cell_prefill(state: MutableMapping[str, Any]) -> None:
    """Prefill комірки з каталогу / фіналістів / черги задач."""
    raw = state.pop(RESEARCH_CELL_PREFILL, None)
    if not isinstance(raw, dict):
        return
    if raw.get("strategy"):
        state["cell_strategy"] = str(raw["strategy"])
    if raw.get("symbol"):
        state["cell_symbol"] = str(raw["symbol"])
    pair = raw.get("pair")
    if not pair and raw.get("leg1") and raw.get("leg2"):
        pair = f"{raw['leg1']}/{raw['leg2']}"
    if pair:
        state["cell_pair"] = str(pair)
        state["cell_symbol"] = str(pair)
    if raw.get("interval"):
        state["cell_interval"] = str(raw["interval"])
    _apply_days(state, raw, "cell_days")


def job_open_target(kind: str, params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Сторінка та оновлення session_state для «Відкрити результат»."""
    combo = combo_from_job_params(kind, params)
    if kind in {"backtest", "pairs", "capacity"}:
        tab = "stress" if kind == "capacity" else "price"
        return "app_pages/cell.py", {RESEARCH_CELL_PREFILL: combo, RESEARCH_CELL_TAB: tab}
    if kind == "overfit":
        return "app_pages/cell.py", {RESEARCH_CELL_PREFILL: combo, RESEARCH_CELL_TAB: "audit"}
    if kind == "sweep":
        return "app_pages/research.py", {RESEARCH_SECTION: "Масовий пошук"}
    return "app_pages/jobs.py", {}


def lookup_job(kind: str, payload: Mapping[str, Any]) -> tuple[Job | None, bool]:
    """Знайти job за fingerprint і чи живий worker."""
    from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore, fingerprint

    with JobStore(DEFAULT_JOBS_PATH) as js:
        job = js.get_by_fingerprint(fingerprint(kind, payload))
        alive = js.worker_is_alive()
    return job, alive


def latest_succeeded_job(
    kind: str,
    *,
    strategy: str,
    symbol: str,
    interval: str,
    days: int | None = None,
    limit: int = 400,
) -> Job | None:
    """Найсвіжіша succeeded-задача для комбінації (list_jobs уже DESC за created_at)."""
    from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore
    from scalper_hft.research.results_table import job_matches_combo

    with JobStore(DEFAULT_JOBS_PATH) as js:
        jobs = js.list_jobs(kind=kind, limit=limit)
    for job in jobs:
        if job.status != "succeeded":
            continue
        if job_matches_combo(job.params, strategy=strategy, symbol=symbol, interval=interval, days=days):
            return job
    return None


def submit_research_job(kind: str, payload: Mapping[str, Any], *, force: bool = False) -> tuple[Job, bool]:
    """Поставити job і повернути (job, worker_alive)."""
    from scalper_hft.research.jobs import DEFAULT_JOBS_PATH, JobStore

    with JobStore(DEFAULT_JOBS_PATH) as js:
        job = js.submit(kind, payload, force=force)
        alive = js.worker_is_alive()
    return job, alive


def job_status_caption(job: Job | None) -> tuple[JobLevel, str]:
    """Рівень і текст статусу задачі для st.info / warning / error."""
    if job is None:
        return "empty", "Ще немає задачі з цими параметрами."
    if job.status in {"queued", "running"}:
        prog = f"{job.progress_done}/{job.progress_total}" if job.progress_total else job.status
        return "info", f"Задача #{job.id} · {job.status} · {prog}"
    if job.status == "failed":
        return "error", f"Задача #{job.id} провалилась: {job.error}"
    if job.status == "cancelled":
        return "warning", f"Задача #{job.id} скасована."
    if job.status == "succeeded":
        return "ok", f"Задача #{job.id} виконана."
    return "warning", f"Задача #{job.id}: {job.status}"
