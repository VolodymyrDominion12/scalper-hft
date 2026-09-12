"""Аудит однієї комірки (стратегія × символ × ТФ): WF + sensitivity + DSR (+ CSCV).

Критерії PASS — overfitting-audit skill:
    avg_oos_sharpe > 0.3, oos_pos_frac >= 0.5, DSR > 0.95 (на OOS-вікнах),
    smoothness > 0.30 (сітка на OOS-регіоні), n_trades_oos >= поріг таймфрейму,
    PBO < 0.5 (якщо CSCV запущено — with_cscv=True).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from scalper_hft.validation.walk_forward import WalkForwardWindow

logger = logging.getLogger(__name__)

VerdictLabel = Literal["PASS", "EXPLORATORY_PASS", "FAIL"]
AuditMode = Literal["exploratory", "final"]
AuditStatus = Literal["ok", "error"]

# Явний holdout для mode="final", коли HOLDOUT_PCT не задано. Без відрізаного
# holdout «фінальний сліпий тест» іде по даних, які вже бачили WF/DSR/sensitivity.
FINAL_HOLDOUT_PCT = 0.20

# Ті самі вікна, що у scripts/matrix_wf_sweeps.sh
WF_TRAIN_TEST: dict[str, tuple[int, int]] = {
    "1m": (4000, 2000),
    "5m": (2000, 1000),
    "15m": (1000, 500),
    "30m": (500, 250),
    "1h": (500, 200),
    "4h": (200, 100),
    # 1d: 3 роки кешу = ~1095 барів. train=200 (~6.6 міс) покриває warm-up
    # найдовших індикаторів набору (EMA-100 / BB-100 / ST lookback-240),
    # test=100 (~3.3 міс) дає 8 OOS-вікон на 3 роки. Без цього запису
    # get(interval, DEFAULT_TRAIN_TEST) підставляв (2000, 500) → на денних
    # даних жодного вікна не існує і ВСІ денні клітинки падали з
    # "Дані коротші за train+purge+test" (status=error), тобто найдешевший
    # за витратами таймфрейм був недоступний для аудиту взагалі.
    "1d": (200, 100),
    "1w": (50, 25),
}
DEFAULT_TRAIN_TEST: tuple[int, int] = (2000, 500)
MIN_TRADES: dict[str, int] = {
    "1m": 100,
    "5m": 60,
    "15m": 40,
    "30m": 40,
    "1h": 30,
    "4h": 20,
    # 1d: за 3 роки трендова система дає 20–60 угод; поріг 4h (20) вимагав би
    # торгувати майже кожен день. 15 угод — статистичний мінімум для OOS-вікон.
    "1d": 15,
    "1w": 8,
}
DEFAULT_MIN_TRADES = 40
OOS_SHARPE_MIN = 0.3
OOS_POS_FRAC_MIN = 0.5
DSR_MIN = 0.95
SMOOTHNESS_MIN = 0.30
PBO_MAX = 0.5
# Видалено DSR_BACKTESTS_PER_COMBO=50 (2026-09-12): «магічний» множник, який
# разом із повним декартовим добутком param_space роздував n_trials для DSR до
# ~10⁶ навіть без жодного підбору параметрів. Див. trial_ledger.effective_n_trials.
CSCV_VARIANTS = 20  # варіантів параметрів для PBO у audit_cell (with_cscv=True)
# AFML Ch.7/11: ненульовий purge/embargo за замовчуванням — 1% OOS-вікна
# (мін. 1 бар). 0 лишає аудит «AFML-shaped, але не AFML-strict»: позиції,
# що перетинають межу train/test, змішують IS і OOS метрики.
DEFAULT_PURGE_EMBARGO_FRAC = 0.01

_SUMMARY_KEYS = (
    "symbol",
    "interval",
    "strategy",
    "status",
    "error",
    "n_windows",
    "avg_is_sharpe",
    "avg_oos_sharpe",
    "oos_pos_frac",
    "degradation",
    "n_trades_oos",
    "sens_param",
    "smoothness",
    "sens_n",
    "sens_error",
    "dsr",
    "n_trials_dsr",
    "pbo",
    "bt_total_return",
    "bt_sharpe",
    "bt_max_dd",
    "bt_n_trades",
    "bt_profit_factor",
    "bt_win_rate",
    "bt_trades_per_day",
    "holdout_sharpe",
    "benchmark_sharpe",
    "quintile_spearman",
    "quintile_pass",
    "time_decay_pass",
    "stress_pass",
)


def default_train_test(interval: str) -> tuple[int, int]:
    """Train/test барів для walk-forward за таймфреймом."""
    return WF_TRAIN_TEST.get(interval, DEFAULT_TRAIN_TEST)


def resolve_wf_windows(
    interval: str,
    train_bars: int | None = None,
    test_bars: int | None = None,
) -> tuple[int, int]:
    """Явні бари, або per-TF дефолт з `default_train_test`.

    None = узяти дефолт інтервалу. Так sweep з 1m і 4h в одній матриці
    не ставить усім train+test=3000 (4h тоді завжди error).
    """
    default_train, default_test = default_train_test(interval)
    train = default_train if train_bars is None else int(train_bars)
    test = default_test if test_bars is None else int(test_bars)
    if train <= 0 or test <= 0:
        raise ValueError(f"train/test мають бути > 0, отримано {train}/{test}")
    return train, test


def min_trades_for(interval: str) -> int:
    """Мінімум угод для статистичної значущості на цьому ТФ."""
    return MIN_TRADES.get(interval, DEFAULT_MIN_TRADES)


def default_purge_embargo(test_bars: int) -> tuple[int, int]:
    """AFML-дефолт purge/embargo для audit_cell: max(1, 1% OOS-вікна).

    Явний 0 у audit_cell вимикає прогін/ембарго (зворотна сумісність для
    відтворення старих прогонів); None = узяти цей дефолт.
    """
    pe = max(1, int(round(test_bars * DEFAULT_PURGE_EMBARGO_FRAC)))
    return pe, pe


def _window_to_dict(w: WalkForwardWindow) -> dict[str, float | int]:
    return {
        "window_idx": w.window_idx,
        "train_start": w.train_start,
        "train_end": w.train_end,
        "test_start": w.test_start,
        "test_end": w.test_end,
        "is_sharpe": float(w.is_sharpe),
        "oos_sharpe": float(w.oos_sharpe),
        "oos_return": float(w.oos_return),
        "n_trades": int(w.n_trades),
    }


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        x = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _as_int(value: object) -> int | None:
    x = _as_float(value)
    if x is None:
        return None
    return int(x)


@dataclass(frozen=True, slots=True)
class CellAudit:
    """Результат аудиту однієї комірки (для JSON / CSV / UI)."""

    symbol: str
    interval: str
    strategy: str
    status: AuditStatus = "error"
    error: str = ""
    n_windows: int = 0
    avg_is_sharpe: float | None = None
    avg_oos_sharpe: float | None = None
    oos_pos_frac: float | None = None
    n_trades_oos: int | None = None
    degradation: float | None = None
    sens_param: str | None = None
    smoothness: float | None = None
    sens_n: int = 0
    sens_error: str | None = None
    dsr: float | None = None
    n_trials_dsr: int | None = None
    pbo: float | None = None  # CSCV PBO (лише якщо audit_cell з with_cscv=True)
    bt_total_return: float | None = None
    bt_sharpe: float | None = None
    bt_max_dd: float | None = None
    bt_n_trades: int | None = None
    bt_profit_factor: float | None = None
    bt_win_rate: float | None = None
    bt_trades_per_day: float | None = None
    holdout_sharpe: float | None = None
    benchmark_sharpe: float | None = None
    quintile_spearman: float | None = None
    quintile_pass: bool | None = None
    time_decay_pass: bool | None = None
    stress_pass: bool | None = None
    windows: tuple[dict[str, float | int], ...] = ()
    sensitivity_grid: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_summary_dict(self) -> dict[str, Any]:
        """Плоский рядок без вікон/сітки (CSV матриці)."""
        d = asdict(self)
        return {k: d[k] for k in _SUMMARY_KEYS}

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CellAudit:
        status_raw = raw.get("status") or "error"
        status: AuditStatus = "ok" if status_raw == "ok" else "error"
        windows = raw.get("windows") or ()
        grid = raw.get("sensitivity_grid") or ()
        n_windows = _as_int(raw.get("n_windows")) or 0
        sens_n = _as_int(raw.get("sens_n")) or 0
        return cls(
            symbol=str(raw.get("symbol") or ""),
            interval=str(raw.get("interval") or ""),
            strategy=str(raw.get("strategy") or ""),
            status=status,
            error=str(raw.get("error") or ""),
            n_windows=n_windows,
            avg_is_sharpe=_as_float(raw.get("avg_is_sharpe")),
            avg_oos_sharpe=_as_float(raw.get("avg_oos_sharpe")),
            oos_pos_frac=_as_float(raw.get("oos_pos_frac")),
            n_trades_oos=_as_int(raw.get("n_trades_oos")),
            degradation=_as_float(raw.get("degradation")),
            sens_param=str(raw["sens_param"]) if raw.get("sens_param") else None,
            smoothness=_as_float(raw.get("smoothness")),
            sens_n=sens_n,
            sens_error=str(raw["sens_error"]) if raw.get("sens_error") else None,
            dsr=_as_float(raw.get("dsr")),
            n_trials_dsr=_as_int(raw.get("n_trials_dsr")),
            pbo=_as_float(raw.get("pbo")),
            bt_total_return=_as_float(raw.get("bt_total_return")),
            bt_sharpe=_as_float(raw.get("bt_sharpe")),
            bt_max_dd=_as_float(raw.get("bt_max_dd")),
            bt_n_trades=_as_int(raw.get("bt_n_trades")),
            bt_profit_factor=_as_float(raw.get("bt_profit_factor")),
            bt_win_rate=_as_float(raw.get("bt_win_rate")),
            bt_trades_per_day=_as_float(raw.get("bt_trades_per_day")),
            holdout_sharpe=_as_float(raw.get("holdout_sharpe")),
            benchmark_sharpe=_as_float(raw.get("benchmark_sharpe")),
            quintile_spearman=_as_float(raw.get("quintile_spearman")),
            quintile_pass=raw.get("quintile_pass") if raw.get("quintile_pass") is not None else None,
            time_decay_pass=raw.get("time_decay_pass") if raw.get("time_decay_pass") is not None else None,
            stress_pass=raw.get("stress_pass") if raw.get("stress_pass") is not None else None,
            windows=tuple(windows),
            sensitivity_grid=tuple(grid),
        )


def cell_verdict(
    audit: CellAudit | Mapping[str, Any],
    *,
    mode: AuditMode = "exploratory",
) -> tuple[VerdictLabel, str]:
    """(PASS/EXPLORATORY_PASS/FAIL, причини) за критеріями overfitting-audit.

    mode=exploratory: критерії пройдені → EXPLORATORY_PASS (не проходить live-гейт).
    mode=final: критерії пройдені → PASS (live-eligible у verdict_store).
    """
    row: Mapping[str, Any] = audit.to_summary_dict() if isinstance(audit, CellAudit) else audit
    interval = str(row.get("interval") or "")
    reasons: list[str] = []

    # Аудит, що впав, не має «проходити» через формальні метрики: раніше
    # status=error повертав лише перелік nan-метрик, і причина збою губилась.
    if row.get("status") is not None and str(row.get("status")) != "ok":
        return "FAIL", f"аудит не виконано: {row.get('error') or 'невідома помилка'}"

    oos = _as_float(row.get("avg_oos_sharpe"))
    if oos is None or oos <= OOS_SHARPE_MIN:
        shown = "nan" if oos is None else f"{oos:.3f}"
        reasons.append(f"avg_oos_sharpe={shown}≤{OOS_SHARPE_MIN}")

    frac = _as_float(row.get("oos_pos_frac"))
    if frac is None or frac < OOS_POS_FRAC_MIN:
        shown = "nan" if frac is None else f"{frac:.0%}"
        reasons.append(f"oos_pos_frac={shown}<{OOS_POS_FRAC_MIN:.0%}")

    dsr = _as_float(row.get("dsr"))
    if dsr is None or dsr <= DSR_MIN:
        shown = "nan" if dsr is None else f"{dsr:.2f}"
        reasons.append(f"DSR={shown}≤{DSR_MIN}")

    sm = _as_float(row.get("smoothness"))
    if sm is None or sm <= SMOOTHNESS_MIN:
        shown = "nan" if sm is None else f"{sm:.2f}"
        reasons.append(f"smoothness={shown}≤{SMOOTHNESS_MIN}")

    # Гейт на OOS-активність (не full-sample бектест): комірка не може
    # пройти з "тонкою" OOS-торгівлею
    nt = _as_int(row.get("n_trades_oos"))
    min_nt = min_trades_for(interval)
    if nt is None or nt < min_nt:
        shown = "nan" if nt is None else str(nt)
        reasons.append(f"n_trades_oos={shown}<{min_nt}")

    # PBO перевіряється лише якщо CSCV дійсно запускався (pbo is not None)
    pbo = _as_float(row.get("pbo"))
    if pbo is not None and pbo > PBO_MAX:
        reasons.append(f"PBO={pbo:.2f}>{PBO_MAX}")

    # Розширені тести (паритет з cmd_report)
    q_pass = row.get("quintile_pass")
    if q_pass is False:
        qs = _as_float(row.get("quintile_spearman"))
        shown = "nan" if qs is None else f"{qs:+.3f}"
        reasons.append(f"quintile_fail ρ={shown}")

    td_pass = row.get("time_decay_pass")
    if td_pass is False:
        reasons.append("time_decay_fail")

    st_pass = row.get("stress_pass")
    if st_pass is False:
        reasons.append("stress_fail")

    bt_sr = _as_float(row.get("bt_sharpe"))
    bench = _as_float(row.get("benchmark_sharpe"))
    if bt_sr is not None and bench is not None and bt_sr <= bench:
        reasons.append(f"bt_sharpe={bt_sr:.3f}≤benchmark={bench:.3f}")

    if mode == "final":
        ho = _as_float(row.get("holdout_sharpe"))
        if ho is None:
            reasons.append("holdout_sharpe=missing")
        elif ho <= 0:
            reasons.append(f"holdout_sharpe={ho:.3f}≤0")

    if not reasons:
        if mode == "final":
            return "PASS", ""
        return "EXPLORATORY_PASS", ""
    return "FAIL", "; ".join(reasons)


def audit_cell(
    strategy_name: str,
    symbol: str,
    interval: str,
    days: int,
    *,
    mode: AuditMode = "exploratory",
    train_bars: int | None = None,
    test_bars: int | None = None,
    with_cscv: bool = False,
    explicit_holdout: bool = False,
    strategy_params: dict[str, Any] | None = None,
    purge_bars: int | None = None,
    embargo_bars: int | None = None,
    n_trials_floor: int | None = None,
    derive: bool = True,
) -> CellAudit:
    """Повний аудит комірки. Помилки даних/рахунку — status=error, без raise.

    mode=exploratory (дефолт): дозволяє EXPLORATORY_PASS у verdict_store;
    live-гейт приймає лише PASS.
    mode=final: вимагає HOLDOUT_PCT>0 або explicit_holdout=True, OOS_ENFORCE_BURN,
    повний CSCV PBO (with_cscv=True); лише PASS у verdict_store.
    with_cscv: додатково рахувати CSCV PBO (~CSCV_VARIANTS додаткових
    бектестів — дорого для матричних прогонів, вмикати для фінального
    вердикту комірки, напр. CLI `overfit`).
    strategy_params: параметри конструктора стратегії (напр. use_kalman).
    purge_bars/embargo_bars: прогін/ембарго між train/test та між OOS-вікнами
        (AFML Ch.7/11) — щоб лейбли/позиції не змішували IS і OOS.
        None = дефолт max(1, 1% test-вікна) (default_purge_embargo);
        явний 0 = вимкнено (лише для відтворення старих прогонів).
    n_trials_floor: мінімальна оцінка числа спроб для DSR (1D). Якщо задано,
        бере max(n_trials_floor, чесна оцінка з журналу/combos) — щоб явна
        вказівка дослідника (--trials) не занижувала DSR-корекцію.
    """

    # Роутер: market_maker аудитується подієвим рушієм, не zero-signal вектором
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.backtest.router import run_strategy_backtest
    from scalper_hft.config import get_settings
    from scalper_hft.data.access import ensure_klines
    from scalper_hft.data.downloader import download_agg_trades, download_funding
    from scalper_hft.strategies import get_strategy
    from scalper_hft.validation.deflated_sharpe import deflated_sharpe_ratio
    from scalper_hft.validation.sensitivity import parameter_sensitivity
    from scalper_hft.validation.walk_forward import run_walk_forward

    try:
        settings = get_settings()
        if mode == "final":
            holdout_pct_cfg = float(getattr(settings, "enforce_holdout_pct", 0.0))
            if holdout_pct_cfg <= 0 and not explicit_holdout:
                return CellAudit(
                    symbol=symbol,
                    interval=interval,
                    strategy=strategy_name,
                    status="error",
                    error="final mode: потрібен HOLDOUT_PCT>0 або explicit_holdout=True",
                )
            if not bool(getattr(settings, "enforce_oos_burn", False)):
                return CellAudit(
                    symbol=symbol,
                    interval=interval,
                    strategy=strategy_name,
                    status="error",
                    error="final mode: потрібен OOS_ENFORCE_BURN=true",
                )
            with_cscv = True
        strategy = get_strategy(strategy_name, **(strategy_params or {}))
        df = ensure_klines(symbol, interval, days, derive=derive)
        if df is None or df.empty:
            return CellAudit(
                symbol=symbol,
                interval=interval,
                strategy=strategy_name,
                status="error",
                error="немає даних",
            )
        cost = CostModel.from_settings(settings, df=df)
        if strategy.needs_trades:
            from scalper_hft.data.access import ensure_trades_coverage

            ensure_trades_coverage(symbol, days)
        trades = download_agg_trades(symbol, days) if strategy.needs_trades else None
        funding = download_funding(symbol, days) if strategy.needs_funding else None

        # «Замкований» holdout (Narang гл. 9): якщо HOLDOUT_PCT > 0, останні
        # holdout_pct% даних НЕ використовуються для WF/sensitivity/DSR/CSCV —
        # лише research-частина (перші 100-holdout_pct %). Holdout лишається
        # недоторканим для фінального сліпого тесту (окремий виклик audit_cell).
        from scalper_hft.validation.holdout import split_research_holdout

        holdout_pct = float(getattr(settings, "enforce_holdout_pct", 0.0))
        if mode == "final" and holdout_pct <= 0:
            # `explicit_holdout=True` дозволяв final при HOLDOUT_PCT=0, і тоді
            # `ho_slice` брався як останні 20% ПОВНОГО df — того самого, що вже
            # пройшов WF/DSR/sensitivity. Тобто «сліпий» тест був не сліпий, а
            # cell_verdict у final перевіряє лише holdout_sharpe > 0.
            # Тепер final завжди відрізає holdout ДО research-прогонів.
            holdout_pct = FINAL_HOLDOUT_PCT
            logger.info(
                "final-режим без HOLDOUT_PCT: застосовано явний holdout %.0f%% "
                "(research-частина обрізається, holdout не бачить WF/DSR/sensitivity)",
                holdout_pct * 100,
            )
        df, holdout_df = split_research_holdout(df, holdout_pct)
        if df.empty:
            return CellAudit(
                symbol=symbol,
                interval=interval,
                strategy=strategy_name,
                status="error",
                error="після holdout-обрізу немає даних (зменшіть HOLDOUT_PCT або збільшіть --days)",
            )

        # OOS-дисципліна (Narang гл. 9): якщо увімкнено OOS_ENFORCE_BURN,
        # перевіряємо, чи цей OOS-відрізок (strategy×symbol×дати) вже
        # «спалений» у реєстрі. Якщо так — fail-closed (status=error). Після
        # успішного аудиту — дописуємо використання, щоб повторний прогін цієї
        # ж комірки вже не міг «випадково» пере-валідуватись на тих самих даних.
        from scalper_hft.validation.oos_registry import check_and_burn

        burn_ok, burn_reason = check_and_burn(
            strategy=strategy_name,
            symbol=symbol,
            df=df,
            days=days,
            purpose=f"audit_cell{'/cscv' if with_cscv else ''}",
            registry_path=getattr(settings, "oos_registry_path", Path("docs/reports/oos_usage.md")),
            enforce=bool(getattr(settings, "enforce_oos_burn", False)),
        )
        if not burn_ok:
            return CellAudit(
                symbol=symbol,
                interval=interval,
                strategy=strategy_name,
                status="error",
                error=burn_reason,
            )

        train, test = default_train_test(interval)
        if train_bars is not None:
            train = int(train_bars)
        if test_bars is not None:
            test = int(test_bars)

        # AFML-дисципліна за замовчуванням: None → ненульовий purge/embargo
        d_purge, d_embargo = default_purge_embargo(test)
        purge_bars = d_purge if purge_bars is None else int(purge_bars)
        embargo_bars = d_embargo if embargo_bars is None else int(embargo_bars)

        wf = run_walk_forward(
            df,
            strategy,
            train,
            test,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
            collect_oos_returns=True,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )
        windows = tuple(_window_to_dict(w) for w in wf.windows)

        # Sensitivity на OOS-регіоні (від початку першого test-вікна):
        # сітка на повній історії забруднена IS-частиною
        df_oos = df.iloc[wf.windows[0].test_start :] if wf.windows else df
        if len(df_oos) < 200:
            df_oos = df

        smoothness: float | None = None
        sens_n = 0
        sens_param: str | None = None
        sens_error: str | None = None
        grid_rows: tuple[dict[str, Any], ...] = ()
        ps = getattr(strategy, "param_space", {}) or {}
        if ps:
            pname = next(iter(ps))
            lo, hi, step = ps[pname]
            step_f = float(step) if step else 1.0
            n_vals = min(int((float(hi) - float(lo)) / step_f) + 1, 15)
            values = [float(lo) + i * step_f for i in range(max(n_vals, 2))][:15]
            try:
                sres = parameter_sensitivity(df_oos, strategy, pname, values, cost=cost, trades=trades, funding=funding)
                smoothness = float(sres.smoothness)
                if smoothness != smoothness:
                    smoothness = None
                sens_n = len(sres.grid)
                sens_param = pname
                grid_rows = tuple({str(k): v for k, v in r.items()} for r in sres.grid.to_dict(orient="records"))
            except Exception as exc:  # noqa: BLE001
                sens_error = str(exc)[:120]

        res_full = run_strategy_backtest(
            df,
            strategy,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
        )
        m = res_full.metrics
        dsr: float | None = None
        n_trials: int | None = None
        # DSR на конкатенованих OOS-дохідностях walk-forward (не full-sample
        # equity — та забруднена IS-вікнами і завищує DSR)
        ret = wf.oos_returns.dropna() if wf.oos_returns is not None else pd.Series(dtype=float)
        if len(ret) >= 2:
            # n_trials = скільки конфігурацій РЕАЛЬНО прогнали, а не розмір
            # теоретичного простору param_space: сам аудит оцінює baseline +
            # сітку sensitivity (sens_n) + CSCV-варіанти, а історію попередніх
            # прогонів дає журнал спроб (усі символи стратегії — вибір символу
            # теж частина пошуку). Деталі й причина зміни — у docstring
            # trial_ledger.effective_n_trials (стара формула combos×50 давала
            # n_trials ~2·10⁶ → DSR>0.95 вимагав річного Sharpe ~4.7).
            variants_this_audit = 1 + int(sens_n) + (CSCV_VARIANTS if with_cscv else 0)
            from scalper_hft.validation.trial_ledger import (
                default_path as _default_ledger_path,
            )
            from scalper_hft.validation.trial_ledger import effective_n_trials, record_trial

            raw_ledger = getattr(settings, "trial_ledger_path", None)
            # Path("") == Path(".") і Path-об'єкти завжди truthy: пусте значення
            # конфігу НЕ можна ловити через `raw_ledger and ...` — інакше журнал
            # "вимкнено" перетворюється на спробу відкрити '.' як файл
            # (IsADirectoryError, валило overfit-аудит без TRIAL_LEDGER_PATH).
            # Якщо TRIAL_LEDGER_PATH не задано — пишемо у дефолтний журнал
            # (інакше множинність спроб ніде не фіксується і DSR знову стає
            # залежним від того, чи налаштував дослідник env-змінну).
            ledger_path = _default_ledger_path()
            if raw_ledger is not None:
                raw_s = str(raw_ledger).strip()
                if raw_s and raw_s != ".":
                    ledger_path = raw_ledger
            n_trials = int(
                effective_n_trials(
                    ledger_path,
                    param_combinations=variants_this_audit,
                    strategy=strategy_name,
                )
            )
            if n_trials_floor is not None and n_trials_floor > 0:
                n_trials = max(n_trials, int(n_trials_floor))
            record_trial(
                ledger_path,
                strategy=strategy_name,
                symbol=symbol,
                purpose=f"audit_cell{'/cscv' if with_cscv else ''}",
                n_trials=variants_this_audit,
                score=float(wf.avg_oos_sharpe),
            )
            dsr = float(deflated_sharpe_ratio(ret.to_numpy(dtype=float), n_trials=n_trials))

        # CSCV PBO (опційно): чи не є IS-кращий варіант перенавченим
        pbo: float | None = None
        if with_cscv and ps:
            from scalper_hft.validation.cscv import pbo_cscv, variant_returns

            try:
                vr = variant_returns(
                    df,
                    strategy,
                    n_variants=CSCV_VARIANTS,
                    cost=cost,
                    trades=trades,
                    funding=funding,
                    position_pct=settings.position_pct,
                )
                pbo = float(pbo_cscv(vr, n_blocks=8, purge_bars=purge_bars, embargo_bars=embargo_bars).pbo)
            except Exception:  # noqa: BLE001
                pbo = None

        from scalper_hft.validation.audit_extensions import run_extended_audit

        ho_slice = holdout_df
        ext = run_extended_audit(
            df,
            strategy,
            cost=cost,
            trades=trades,
            funding=funding,
            position_pct=settings.position_pct,
            bar_returns=res_full.bar_returns,
            baseline_max_dd=float(m.max_drawdown),
            holdout_df=ho_slice if mode == "final" or holdout_pct > 0 else None,
            strategy_params=strategy_params or {},
        )

        return CellAudit(
            symbol=symbol,
            interval=interval,
            strategy=strategy_name,
            status="ok",
            n_windows=len(wf.windows),
            avg_is_sharpe=float(wf.avg_is_sharpe),
            avg_oos_sharpe=float(wf.avg_oos_sharpe),
            oos_pos_frac=float(wf.positive_windows_frac),
            n_trades_oos=int(sum(w.n_trades for w in wf.windows)),
            degradation=float(wf.degradation),
            sens_param=sens_param,
            smoothness=smoothness,
            sens_n=sens_n,
            sens_error=sens_error,
            dsr=dsr,
            n_trials_dsr=n_trials,
            pbo=pbo,
            bt_total_return=float(m.total_return),
            bt_sharpe=float(m.sharpe),
            bt_max_dd=float(m.max_drawdown),
            bt_n_trades=int(m.n_trades),
            bt_profit_factor=float(m.profit_factor),
            bt_win_rate=float(m.win_rate),
            bt_trades_per_day=float(m.trades_per_day),
            holdout_sharpe=ext.holdout_sharpe,
            benchmark_sharpe=ext.benchmark_sharpe,
            quintile_spearman=ext.quintile.spearman if ext.quintile else None,
            quintile_pass=ext.quintile.pass_ if ext.quintile else None,
            time_decay_pass=ext.time_decay.pass_ if ext.time_decay else None,
            stress_pass=ext.stress.pass_ if ext.stress else None,
            windows=windows,
            sensitivity_grid=grid_rows,
        )
    except Exception as exc:  # noqa: BLE001
        return CellAudit(
            symbol=symbol,
            interval=interval,
            strategy=strategy_name,
            status="error",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
