"""MCP-сервер для трейдінгу scalper-hft (Model Context Protocol, stdio).

Дозволяє AI-асистенту (Claude Code / Cursor / DSH) викликати можливості
проєкту як інструменти: бектест, cohort/stress/capacity аналіз, список
стратегій, стан кешу даних, один крок paper-торгівлі.

Транспорт: MCP stdio — JSON-RPC 2.0, повідомлення рядками (newline-delimited)
на stdin/stdout. Реалізація на чистому stdlib (SDK `mcp` не потрібен).

⚠ Безпека: НІКОЛИ не повертаємо API-ключі. `settings_summary` маскує секрети;
всі виклики — dry-run/read-only (paper-крок не відправляє реальні ордери).

Запуск:
    .venv/bin/python -m scalper_hft.mcp_trading          # як MCP-сервер
    .venv/bin/python -m scalper_hft.cli mcp              # те саме через CLI
"""

from __future__ import annotations

import json
import sys
from typing import Any

# ── Інструменти ──────────────────────────────────────────────────────────────

def _strategy_list(args: dict | None = None) -> dict:
    from scalper_hft.strategies import REGISTRY

    return {
        "strategies": [
            {"name": name, "param_space": {k: list(v) for k, v in cls.param_space.items()}}
            for name, cls in sorted(REGISTRY.items())
        ]
    }


def _settings_summary(args: dict | None = None) -> dict:
    from scalper_hft.config import get_settings

    s = get_settings()
    return {
        "dry_run": s.dry_run,
        "exchange": s.exchange,
        "maker_fee": s.maker_fee,
        "taker_fee": s.taker_fee,
        "slippage_bps": s.slippage_bps,
        "maker_execution": s.maker_execution,
        "position_pct": s.position_pct,
        "daily_loss_limit": s.daily_loss_limit,
        "max_consecutive_losses": s.max_consecutive_losses,
        "default_symbols": list(s.default_symbols),
        # ключі НЕ повертаються
        "api_key_configured": bool(s.binance_api_key),
    }


def _market_status(args: dict | None = None) -> dict:
    from scalper_hft.config import get_settings

    data_dir = get_settings().data_dir_abs
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]
    rows = []
    for sym in symbols:
        k = data_dir / f"{sym}_1m_klines.parquet"
        k5 = data_dir / f"{sym}_5m_klines.parquet"
        t = data_dir / f"{sym}_aggTrades.parquet"
        f = data_dir / f"{sym}_funding.parquet"
        rows.append(
            {
                "symbol": sym,
                "klines_1m": k.stat().st_size if k.exists() else 0,
                "klines_5m": k5.stat().st_size if k5.exists() else 0,
                "aggTrades": t.stat().st_size if t.exists() else 0,
                "funding": f.stat().st_size if f.exists() else 0,
            }
        )
    return {"symbols": rows}


def _load_df(symbol: str, interval: str, days: int):
    from scalper_hft.cli import _load_klines

    return _load_klines(symbol, interval, days)


def _run_backtest(args: dict) -> dict:
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy

    symbol = str(args.get("symbol", "BTCUSDT"))
    interval = str(args.get("interval", "5m"))
    days = int(args.get("days", 30))
    strategy = get_strategy(str(args.get("strategy", "mean_reversion")), **dict(args.get("params", {})))
    df = _load_df(symbol, interval, days)
    s = get_settings()
    cost = CostModel(maker_fee=s.maker_fee, taker_fee=s.taker_fee, slippage_frac=s.slippage_frac)
    trades = None
    if getattr(strategy, "needs_trades", False):
        from scalper_hft.data.downloader import download_agg_trades

        trades = download_agg_trades(symbol, days)
    funding = None
    if getattr(strategy, "needs_funding", False):
        from scalper_hft.data.downloader import download_funding

        funding = download_funding(symbol, days)
    res = run_backtest(df, strategy, cost=cost, trades=trades, funding=funding, position_pct=s.position_pct)
    m = res.metrics
    return {
        "strategy": strategy.name,
        "symbol": symbol,
        "interval": interval,
        "bars": len(df),
        "total_return": m.total_return,
        "sharpe": m.sharpe,
        "sharpe_hourly": m.sharpe_hourly,
        "max_drawdown": m.max_drawdown,
        "win_rate": m.win_rate,
        "n_trades": m.n_trades,
        "trades_per_day": m.trades_per_day,
    }


def _run_analysis(kind: str, args: dict) -> dict:
    """cohort / stress / capacity аналіз з реальних даних."""
    from scalper_hft.backtest.engine import run_backtest
    from scalper_hft.backtest.execution import CostModel
    from scalper_hft.config import get_settings
    from scalper_hft.strategies import get_strategy

    symbol = str(args.get("symbol", "BTCUSDT"))
    interval = str(args.get("interval", "5m"))
    days = int(args.get("days", 60))
    strategy = get_strategy(str(args.get("strategy", "mean_reversion")), **dict(args.get("params", {})))
    df = _load_df(symbol, interval, days)
    s = get_settings()
    cost = CostModel(maker_fee=s.maker_fee, taker_fee=s.taker_fee, slippage_frac=s.slippage_frac)
    res = run_backtest(df, strategy, cost=cost, position_pct=s.position_pct)
    ret = res.equity.pct_change().dropna()

    if kind == "cohort":
        from scalper_hft.validation.cohort import cohort_report

        return {"report": cohort_report(res.trades)}
    if kind == "stress":
        from scalper_hft.validation.stress import stress_report

        rep = stress_report(ret)
        return {"scenarios": rep.round(4).to_dict("index")}
    if kind == "capacity":
        from scalper_hft.validation.capacity import capacity_curve, saturation_scale

        scales = [float(x) for x in args.get("scales", [1.0, 2.0, 5.0, 10.0])]
        curve = capacity_curve(df, strategy, scales=scales, cost=cost, position_pct=s.position_pct)
        return {
            "curve": curve.round(4).to_dict("records"),
            "saturation_scale": saturation_scale(curve),
        }
    raise ValueError(f"невідомий аналіз: {kind}")


def _paper_step(args: dict) -> dict:
    """Один крок paper-торгівлі (dry-run, без реальних ордерів)."""
    from scalper_hft.data.downloader import download_klines
    from scalper_hft.live.trader import LiveTrader, run_trader_once
    from scalper_hft.strategies import get_strategy

    symbol = str(args.get("symbol", "BTCUSDT"))
    interval = str(args.get("interval", "5m"))
    strategy = get_strategy(str(args.get("strategy", "mean_reversion")), **dict(args.get("params", {})))
    df = download_klines(symbol, interval, days=int(args.get("days", 7)))
    if df is None or df.empty:
        raise ValueError(f"немає даних {symbol} {interval}")
    trader = LiveTrader(strategy, symbol, interval)
    action = run_trader_once(trader, df)
    return {
        "action": action,
        "equity": trader.account.equity,
        "open_positions": len(trader.account.positions),
        "trades": len(trader.account.trades),
    }


TOOLS: dict[str, dict[str, Any]] = {
    "strategy_list": {
        "description": "Список стратегій з реєстру та їхні параметри.",
        "handler": _strategy_list,
    },
    "settings_summary": {
        "description": "Поточні налаштування (без секретів).",
        "handler": _settings_summary,
    },
    "market_status": {
        "description": "Розмір кешу даних по символах (klines/aggTrades/funding).",
        "handler": _market_status,
    },
    "run_backtest": {
        "description": "Бектест стратегії на кешованих даних. Args: strategy, symbol, interval, days, params.",
        "handler": _run_backtest,
    },
    "run_cohort": {
        "description": "Cohort decay-аналіз угод стратегії.",
        "handler": lambda a: _run_analysis("cohort", a),
    },
    "run_stress": {
        "description": "Стрес-тест стратегії (crash/liquidity/vol_spike/funding_shock).",
        "handler": lambda a: _run_analysis("stress", a),
    },
    "run_capacity": {
        "description": "Capacity-тест: Sharpe при масштабуванні позицій.",
        "handler": lambda a: _run_analysis("capacity", a),
    },
    "paper_step": {
        "description": "Один крок paper-торгівлі на останньому закритому барі (dry-run).",
        "handler": _paper_step,
    },
}


# ── MCP stdio-транспорт (JSON-RPC 2.0, newline-delimited) ────────────────────

_PROTOCOL_VERSION = "2024-11-05"


def _result(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def handle_message(msg: dict) -> list[dict]:
    """Обробляє одне JSON-RPC повідомлення. Повертає список відповідей."""
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        return [_result(msg_id, {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "scalper-hft", "version": "0.1.0"},
        })]
    if method == "notifications/initialized":
        return []  # notification — без відповіді
    if method == "ping":
        return [_result(msg_id, {})]
    if method == "tools/list":
        tools = [
            {"name": name, "description": spec["description"], "inputSchema": {"type": "object"}}
            for name, spec in TOOLS.items()
        ]
        return [_result(msg_id, {"tools": tools})]
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        spec = TOOLS.get(name)
        if spec is None:
            return [_error(msg_id, -32602, f"невідомий інструмент: {name}")]
        try:
            out = spec["handler"](args)
            text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
            return [_result(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})]
        except Exception as exc:  # noqa: BLE001
            return [_result(msg_id, {"content": [{"type": "text", "text": f"помилка: {exc}"}], "isError": True})]
    return [_error(msg_id, -32601, f"невідомий метод: {method}")]


def run_stdio() -> None:
    """Головний цикл MCP-сервера: читає рядки JSON з stdin, пише у stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        for resp in handle_message(msg):
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def main() -> None:
    run_stdio()


if __name__ == "__main__":
    main()
