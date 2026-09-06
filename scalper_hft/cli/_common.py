"""Спільні хелпери CLI (розбиття монолітного cli.py на пакет)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from scalper_hft.config import get_settings

logger = logging.getLogger("scalper_hft.cli")

_SHORT_PAIRS_INTERVALS = frozenset({"1m", "5m"})


def fail(msg: str, code: int = 1) -> None:
    """Єдина точка помилок CLI: лог + SystemExit (замість міксу sys.exit/raise)."""
    logger.error("%s", msg)
    raise SystemExit(code)


def _warn_pairs_short_interval(strategy: str, interval: str | None) -> None:
    if strategy == "pairs_arb" and str(interval) in _SHORT_PAIRS_INTERVALS:
        logger.warning(
            "pairs_arb на %s: fee-drag уже відхилив 1m; валідований edge — 1h maker",
            interval,
        )


def _apply_use_kalman(args: argparse.Namespace, params: dict) -> dict:
    out = dict(params)
    if getattr(args, "use_kalman", False):
        out["use_kalman"] = True
    return out


def _add_overlay_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--overlay",
        nargs="?",
        const="configs/cells/default.yaml",
        default=None,
        help="Cell Overlay YAML. Без шляху — configs/cells/default.yaml. Без прапорця шар вимкнено.",
    )


def _policy_from_args(args: argparse.Namespace):
    """None якщо --overlay не задано."""
    path = getattr(args, "overlay", None)
    if not path:
        return None
    from scalper_hft.overlay import load_overlay_book, resolve_policy

    settings = get_settings()
    symbol = args.symbol or (settings.default_symbols[0] if settings.default_symbols else "BTCUSDT")
    interval = args.interval or settings.default_interval
    book = load_overlay_book(path)
    return resolve_policy(book, args.strategy, symbol, interval)


def _merge_overlay_params(args: argparse.Namespace, params: dict, overlay) -> dict:
    from scalper_hft.overlay.resolver import bind_strategy_kwargs
    from scalper_hft.strategies import REGISTRY

    bound = bind_strategy_kwargs(REGISTRY[args.strategy], overlay.strategy_kwargs())
    for key, val in bound.items():
        params.setdefault(key, val)
    return params


def _load_klines(
    symbol: str, interval: str, days: int, base: str | None = None, derive: bool = True, exchange_id: str | None = None
) -> pd.DataFrame:
    """Хелпер для CLI: завантажити/ресемплити дані або впасти з помилкою."""
    from scalper_hft.data.access import ensure_klines

    df = ensure_klines(symbol, interval, days, base_interval=base or "1m", derive=derive, exchange_id=exchange_id)
    if df is None or df.empty:
        logger.error("Немає даних %s %s — запустіть download спершу", symbol, interval)
        sys.exit(1)
    return df


def _param_combinations(strategy) -> int:
    """Кількість комбінацій параметрів у param_space (добуток розмірів ґраток).

    Раніше в DSR передавався `len(param_space)` (кількість ПАРАМЕТРІВ, напр. 3),
    а не кількість спроб/комбінацій → корекція на множинне тестування була
    занижена в рази. Тут — добуток кількості значень по кожному параметру
    (обрізаний зверху, щоб не вибухало).
    """
    ps = getattr(strategy, "param_space", {}) or {}
    if not ps:
        return 1
    combos = 1
    for lo, hi, step in ps.values():
        step = float(step) if step else 1.0
        n_vals = max(int((float(hi) - float(lo)) / step) + 1, 1)
        combos *= n_vals
    return min(max(combos, 1), 100_000)


def _enqueue_job(kind: str, params: dict, *, force: bool = False) -> None:
    from scalper_hft.research.jobs import JobStore

    with JobStore() as store:
        job = store.submit(kind, params, force=force)
        print(f"job id={job.id} kind={job.kind} status={job.status} fp={job.short_fp}")
        if job.status == "succeeded" and not force:
            print(f"уже виконано з цими параметрами; повтор — job rerun {job.id}")
        if not store.worker_is_alive():
            print("воркер не запущений: uv run python -m scalper_hft.cli job worker --jobs 2")


def _plot_equity(equity: pd.Series, strategy: str, symbol: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        equity.plot(ax=ax, title=f"{strategy} · {symbol}")
        ax.set_ylabel("Equity")
        out = Path("docs/plots")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{strategy}_{symbol}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        logger.info("Графік збережено: %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не вдалося побудувати графік: %s", exc)


def _parse_param_dict(args: list[str]) -> dict:
    out: dict = {}
    for item in args or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def _api_bind(
    host: str | None,
    port: int | None,
    *,
    default_host: str,
    default_port: int,
) -> tuple[str, int]:
    """CLI `--host`/`--port` override settings; omit them to keep API_HOST/API_PORT."""
    bind_host = host or default_host
    bind_port = default_port if port is None else port
    if not 1 <= bind_port <= 65535:
        raise ValueError(f"API port must be 1-65535, got {bind_port}")
    return bind_host, bind_port
