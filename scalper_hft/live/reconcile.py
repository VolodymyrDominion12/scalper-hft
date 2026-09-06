"""Звірка локальних позицій з біржею; kill-switch при розходженні."""

from __future__ import annotations

from dataclasses import dataclass

from scalper_hft.live.account import PaperAccount


class KillSwitch(RuntimeError):
    """Локальний стан розійшовся з біржею — торгівлю зупинено."""


@dataclass(frozen=True)
class ExchangePosition:
    symbol: str
    side: str  # long | short | flat
    size: float


def parse_exchange_positions(raw: list[dict]) -> dict[str, ExchangePosition]:
    """Нормалізує відповідь ccxt fetch_positions."""
    out: dict[str, ExchangePosition] = {}
    for row in raw:
        symbol = str(row.get("symbol") or row.get("info", {}).get("symbol") or "")
        if not symbol:
            continue
        symbol = symbol.replace("/", "").replace(":USDT", "")
        contracts = float(row.get("contracts") or row.get("size") or 0.0)
        side_raw = str(row.get("side") or "").lower()
        if contracts == 0 or side_raw in {"", "flat"}:
            continue
        side = "long" if side_raw in {"long", "buy"} else "short"
        out[symbol] = ExchangePosition(symbol, side, abs(contracts))
    return out


def reconcile_positions(
    account: PaperAccount,
    exchange: dict[str, ExchangePosition],
    *,
    size_tol: float = 1e-8,
    scope: set[str] | None = None,
) -> tuple[bool, str]:
    """Порівняти локальні ноги з біржею. Ключі account можуть бути `pair:SYMBOL`.

    scope: множина символів, які КЕРУЄ цей трейдер/ранере. Якщо задано —
    сторонні позиції того ж біржового рахунку (інший трейдер, ручні угоди)
    НЕ тригерять KillSwitch: звіряються лише локальні ноги в межах scope і
    лише exchange-позиції в межах scope. Без scope поведінка стара (exact-set).
    """
    local_net: dict[str, float] = {}
    for key, pos in account.positions.items():
        sym = key.split(":")[-1]
        if scope is not None and sym not in scope:
            continue
        signed = pos.size if pos.side == "long" else -pos.size
        local_net[sym] = local_net.get(sym, 0.0) + signed

    local: dict[str, tuple[str, float]] = {}
    for sym, net_val in local_net.items():
        if abs(net_val) > size_tol:
            side = "long" if net_val > 0 else "short"
            local[sym] = (side, abs(net_val))

    for sym, (side, size) in local.items():
        ex = exchange.get(sym)
        if ex is None:
            return False, f"локально є {sym} {side}, на біржі немає"
        if ex.side != side:
            return False, f"{sym}: локально {side}, біржа {ex.side}"
        if abs(ex.size - size) > size_tol * max(size, 1.0):
            return False, f"{sym}: розмір {size} vs біржа {ex.size}"

    exchange_filtered = {s: ex for s, ex in exchange.items() if scope is None or s in scope}
    for sym, ex in exchange_filtered.items():
        if sym not in local:
            return False, f"на біржі є {sym} {ex.side}, локально немає"
    return True, "ok"


def halt_if_drift(
    account: PaperAccount,
    raw_positions: list[dict],
    *,
    dry_run: bool = True,
    scope: set[str] | None = None,
) -> None:
    """У live: кинути KillSwitch при розходженні. Paper — no-op."""
    if dry_run:
        return
    ok, reason = reconcile_positions(account, parse_exchange_positions(raw_positions), scope=scope)
    if not ok:
        raise KillSwitch(reason)


def reconcile_exchange_state(
    account: PaperAccount,
    client: object | None,
    *,
    dry_run: bool,
    scope: set[str] | None = None,
) -> None:
    """Paper: no-op. Live: fetch_positions + halt_if_drift. KillSwitch не ковтати."""
    if dry_run:
        return
    if client is None or not hasattr(client, "fetch_positions"):
        raise RuntimeError("Live звірка потребує клієнт біржі з fetch_positions")
    raw = client.fetch_positions()
    halt_if_drift(account, raw, dry_run=False, scope=scope)
