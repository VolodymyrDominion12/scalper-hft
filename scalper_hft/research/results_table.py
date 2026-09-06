"""Метадані колонок sweep/walk-forward і фільтрація таблиць результатів.

Без Streamlit: сторінки дашборду будують column_config з цих підказок,
а тести перевіряють фільтри без UI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

FormatKind = Literal["text", "int", "float", "percent", "datetime"]


@dataclass(frozen=True, slots=True)
class ColumnHint:
    """Підпис, формат і пояснення колонки для таблиці результатів."""

    key: str
    label: str
    help: str
    kind: FormatKind = "float"
    number_format: str | None = None
    pinned: bool = False


SWEEP_COLUMNS: tuple[ColumnHint, ...] = (
    ColumnHint("strategy", "Стратегія", "Ім'я альфа-моделі з реєстру стратегій.", "text", pinned=True),
    ColumnHint("symbol", "Символ", "Інструмент USDT-M (одна нога). Пари в цей sweep не входять.", "text"),
    ColumnHint("interval", "ТФ", "Таймфрейм барів, на якому рахувався сигнал.", "text"),
    ColumnHint(
        "mode",
        "Режим",
        "**backtest** — один прогін in-sample. **walkforward** — середнє по OOS-вікнах; "
        "колонка Sharpe тоді = середній OOS Sharpe.",
        "text",
    ),
    ColumnHint("days", "Днів", "Глибина історії, на якій рахувалась клітинка.", "int", "%d"),
    ColumnHint("n_bars", "Барів", "Скільки барів увійшло в прогін після ресемплу.", "int", "%d"),
    ColumnHint("n_trades", "Угод", "Кількість закритих угод (для walk-forward — сума по вікнах).", "int", "%d"),
    ColumnHint(
        "total_return",
        "Дохідність",
        "Сумарна відносна зміна equity за період (після комісій і slippage). Порожньо у walk-forward.",
        "percent",
        "%.2%",
    ),
    ColumnHint(
        "sharpe",
        "Sharpe",
        "Річний Sharpe з барних returns. У режимі walkforward сюди пишеться середній **OOS** Sharpe.",
        "float",
        "%.3f",
    ),
    ColumnHint("sortino", "Sortino", "Як Sharpe, але в знаменнику лише негативна волатильність.", "float", "%.3f"),
    ColumnHint(
        "calmar", "Calmar", "CAGR / |max drawdown|. Високий Calmar = дохідність без глибоких ям.", "float", "%.3f"
    ),
    ColumnHint("max_dd", "Max DD", "Найглибша просадка equity від піку (від'ємна частка).", "percent", "%.2%"),
    ColumnHint("win_rate", "Win rate", "Частка угод з додатним PnL після комісій.", "percent", "%.0%"),
    ColumnHint("profit_factor", "PF", "Сума прибутків / сума збитків. < 1 — стратегія в мінусі.", "float", "%.2f"),
    ColumnHint("avg_trade", "Avg trade", "Середній PnL однієї угоди (відносно номіналу).", "percent", "%.3%"),
    ColumnHint("exposure", "Exposure", "Частка часу в позиції (не у кеші).", "percent", "%.0%"),
    ColumnHint(
        "trades_per_day", "Угод/день", "Частота угод. Занадто мало — шум; занадто багато — комісії.", "float", "%.2f"
    ),
    ColumnHint(
        "avg_is_sharpe",
        "IS Sharpe",
        "Середній in-sample Sharpe по вікнах walk-forward (не для висновку про edge).",
        "float",
        "%.3f",
    ),
    ColumnHint(
        "avg_oos_sharpe",
        "OOS Sharpe",
        "Середній Sharpe на тестових вікнах walk-forward. Головна метрика sweep.",
        "float",
        "%.3f",
    ),
    ColumnHint(
        "oos_positive_frac", "OOS+", "Частка OOS-вікон з додатним Sharpe. Поріг аудиту ≥ 50%.", "percent", "%.0%"
    ),
    ColumnHint("n_raw_signals", "Raw сигналів", "Сигналів до фільтрів (якщо у sweep увімкнено tracing).", "int", "%d"),
    ColumnHint("n_filtered", "Зрізано", "Скільки raw-сигналів заблокували фільтри.", "int", "%d"),
    ColumnHint("status", "Статус", "**ok** — клітинка порахована. **error** — дивіться колонку помилки.", "text"),
    ColumnHint("error", "Помилка", "Чому клітинка не порахувалась (немає даних, виняток стратегії).", "text"),
    ColumnHint("run_ts", "Прогін", "Час запису клітинки в sweep.db (UTC).", "datetime"),
)

SWEEP_COLUMN_BY_KEY: dict[str, ColumnHint] = {c.key: c for c in SWEEP_COLUMNS}

SWEEP_DISPLAY_KEYS: tuple[str, ...] = (
    "strategy",
    "symbol",
    "interval",
    "mode",
    "days",
    "n_trades",
    "total_return",
    "sharpe",
    "sortino",
    "calmar",
    "max_dd",
    "win_rate",
    "profit_factor",
    "trades_per_day",
    "avg_oos_sharpe",
    "oos_positive_frac",
    "avg_is_sharpe",
    "status",
    "error",
)

TRADE_COLUMNS: tuple[ColumnHint, ...] = (
    ColumnHint(
        "Вхід", "Вхід", "Час заповнення входу (бар t+1 після сигналу на закритті t).", "datetime", "DD.MM.YYYY HH:mm"
    ),
    ColumnHint(
        "Вихід", "Вихід", "Час закриття позиції (сигнал 0 / SL / TP / кінець даних).", "datetime", "DD.MM.YYYY HH:mm"
    ),
    ColumnHint("Сторона", "Сторона", "Лонг = +1, шорт = −1.", "text"),
    ColumnHint(
        "Ціна входу",
        "Ціна входу",
        "Ціна виконання входу (close бара виконання + slippage/комісія в PnL).",
        "float",
        "%.4f",
    ),
    ColumnHint("Ціна виходу", "Ціна виходу", "Ціна виконання виходу.", "float", "%.4f"),
    ColumnHint("SL", "SL", "Stop-loss з `Strategy.exit_levels`, якщо стратегія його задає.", "float", "%.4f"),
    ColumnHint("TP", "TP", "Take-profit з `Strategy.exit_levels`, якщо стратегія його задає.", "float", "%.4f"),
    ColumnHint("PnL, %", "PnL, %", "Дохідність угоди в відсотках після комісій і slippage.", "float", "%.3f"),
    ColumnHint(
        "PnL, $", "PnL, $", "Та сама дохідність × стартовий капітал таблиці (за замовчуванням 10 000).", "float", "%.2f"
    ),
)

_NUMERIC_SWEEP: tuple[str, ...] = tuple(c.key for c in SWEEP_COLUMNS if c.kind != "text" and c.kind != "datetime")

OPEN_DETAILS_LABEL = ":material/candlestick_chart: Деталі"
OPEN_AUDIT_LABEL = ":material/fact_check: Аудит"


def coerce_sweep_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Привести метрики sweep до чисел (SQLite інколи віддає TEXT)."""
    out = df.copy()
    for col in _NUMERIC_SWEEP:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def display_columns(df: pd.DataFrame) -> list[str]:
    """Колонки для таблиці: канонічний порядок, лише ті, що є в df."""
    return [c for c in SWEEP_DISPLAY_KEYS if c in df.columns]


def default_sort_column(df: pd.DataFrame) -> str:
    """OOS Sharpe, якщо є хоч одне значення; інакше in-sample Sharpe."""
    if "avg_oos_sharpe" in df.columns and df["avg_oos_sharpe"].notna().any():
        return "avg_oos_sharpe"
    if "sharpe" in df.columns:
        return "sharpe"
    if "n_trades" in df.columns:
        return "n_trades"
    return df.columns[0] if len(df.columns) else "strategy"


def filter_sweep_results(
    df: pd.DataFrame,
    *,
    strategies: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    intervals: Sequence[str] | None = None,
    mode: str | None = None,
    min_trades: float | None = None,
    min_sharpe: float | None = None,
    min_win_rate: float | None = None,
    query: str = "",
    ok_only: bool = True,
) -> pd.DataFrame:
    """Дешеві інтерактивні фільтри поверх уже завантаженого sweep DataFrame."""
    if df.empty:
        return df
    view = coerce_sweep_metrics(df)
    if ok_only and "status" in view.columns:
        view = view[view["status"].fillna("ok") == "ok"]
    if strategies:
        view = view[view["strategy"].isin(list(strategies))]
    if symbols:
        view = view[view["symbol"].isin(list(symbols))]
    if intervals:
        view = view[view["interval"].isin(list(intervals))]
    if mode and mode not in {"", "Усі", "all"} and "mode" in view.columns:
        view = view[view["mode"] == mode]
    if min_trades is not None and "n_trades" in view.columns:
        view = view[view["n_trades"].fillna(0) >= float(min_trades)]
    if min_sharpe is not None and "sharpe" in view.columns:
        view = view[view["sharpe"].fillna(float("-inf")) >= float(min_sharpe)]
    if min_win_rate is not None and "win_rate" in view.columns:
        view = view[view["win_rate"].fillna(0) >= float(min_win_rate)]
    q = query.strip().lower()
    if q:
        hay = pd.Series("", index=view.index, dtype="object")
        for col in ("strategy", "symbol", "interval", "mode", "error"):
            if col in view.columns:
                hay = hay + " " + view[col].astype(str).str.lower()
        view = view[hay.str.contains(q, regex=False)]
    return view.reset_index(drop=True)


def row_actions(mode: str | None) -> list[str]:
    """Кнопки рядка: завжди деталі бектесту; для walk-forward ще аудит."""
    actions = [OPEN_DETAILS_LABEL]
    if str(mode or "") == "walkforward":
        actions.append(OPEN_AUDIT_LABEL)
    return actions


def is_audit_action(label: str) -> bool:
    return "Аудит" in label


def job_matches_combo(
    params: Mapping[str, Any],
    *,
    strategy: str,
    symbol: str,
    interval: str,
    days: int | None = None,
) -> bool:
    """Чи params задачі збігаються зі стратегією / символом / ТФ (і опційно днів)."""
    if str(params.get("strategy") or "") != strategy:
        return False
    if str(params.get("symbol") or "") != symbol:
        return False
    if str(params.get("interval") or "") != interval:
        return False
    if days is None:
        return True
    try:
        return int(params.get("days") or -1) == int(days)
    except (TypeError, ValueError):
        return False


def trade_quality_blockers(
    *,
    job_status: str | None,
    has_artifacts: bool,
    n_trades: int,
    has_klines: bool,
) -> list[str]:
    """Людські причини, чому блок «Якість угод» порожній."""
    reasons: list[str] = []
    if job_status is None:
        reasons.append(
            "Немає повного бектесту з цими параметрами. Sweep зберігає лише підсумок метрик, без списку угод."
        )
    elif job_status in {"queued", "running"}:
        reasons.append(f"Бектест ще рахується ({job_status}). Дочекайтесь succeeded у «Задачах».")
    elif job_status == "failed":
        reasons.append("Бектест провалився — відкрийте лог на сторінці «Задачі».")
    elif job_status == "cancelled":
        reasons.append("Бектест скасовано. Поставте задачу знову.")
    elif job_status != "succeeded":
        reasons.append(f"Бектест у статусі {job_status}.")
    elif not has_artifacts:
        reasons.append("Задача succeeded, але parquet-артефакти ще не записані.")
    elif n_trades <= 0:
        reasons.append("У цьому бектесті немає закритих угод — MAE/MFE і heatmap порожні.")
    if not has_klines:
        reasons.append("Немає свічок у кеші для цього символу/ТФ — MAE/MFE потребує high/low.")
    return reasons
