"""FastAPI сервер: REST API та WebSocket для дашборду."""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scalper_hft.api.auth import require_api_capability, verify_token
from scalper_hft.api.broadcaster import manager
from scalper_hft.config import get_settings
from scalper_hft.live.control import load_control, save_control
from scalper_hft.live.store import PaperStore

logger = logging.getLogger(__name__)

_broadcast_task: asyncio.Task | None = None


class ControlUpdateRequest(BaseModel):
    pause: bool | None = Field(default=None, description="Призупинити торгівлю")
    no_new_entries: bool | None = Field(default=None, description="Заборонити відкриття нових позицій")
    flatten: bool | None = Field(default=None, description="Закрити всі відкриті позиції")


class ClosePositionRequest(BaseModel):
    symbol: str = Field(..., description="Символ для закриття, наприклад BTCUSDT")
    exchange: str = Field(default="binance", description="Біржа")
    mode: str = Field(default="paper", description="paper або live")


class MockPositionRequest(BaseModel):
    symbol: str = Field(..., description="Символ, наприклад BTCUSDT")
    side: str = Field(default="long", description="long або short")
    size: float = Field(default=0.1, description="Розмір позиції")
    entry_price: float = Field(default=50000.0, description="Ціна входу")
    unrealized_pnl: float = Field(default=15.5, description="Нереалізований PnL")
    exchange: str = Field(default="binance", description="Біржа")
    mode: str = Field(default="paper", description="paper або live")


def _extract_positions(store: PaperStore) -> list[dict[str, Any]]:
    pos_df = store.open_positions()
    positions: list[dict[str, Any]] = []
    if pos_df is not None and not pos_df.empty:
        for _, p in pos_df.iterrows():
            positions.append(
                {
                    "id": p.get("id") if hasattr(p, "get") else None,
                    "symbol": str(p.get("symbol", "")),
                    "side": str(p.get("side", "flat")),
                    "size": float(p.get("size", 0.0) or 0.0),
                    "entry_price": float(p.get("entry_price", 0.0) or 0.0),
                    "mark_price": float(p.get("mark_price", 0.0) or 0.0) if pd.notna(p.get("mark_price")) else 0.0,
                    "unrealized_pnl": float(p.get("unrealized_pnl", 0.0) or 0.0)
                    if pd.notna(p.get("unrealized_pnl"))
                    else 0.0,
                    "exchange": str(p.get("exchange", "binance")),
                    "mode": str(p.get("mode", "paper")),
                    "ts": str(p.get("ts", "")),
                }
            )
    return positions


def _extract_accounts(store: PaperStore) -> list[dict[str, Any]]:
    acc_df = store.recent_accounts(limit=10)
    accounts = []
    if hasattr(acc_df, "iterrows"):
        for _, a in acc_df.iterrows():
            accounts.append(
                {
                    "exchange": a["exchange"],
                    "mode": a["mode"],
                    "balance": float(a["balance"]),
                    "unrealized_pnl": float(a["unrealized_pnl"]),
                    "margin_used": float(a["margin_used"]),
                    "available": float(a["available"]),
                    "ts": str(a["ts"]),
                }
            )
    return accounts


def _extract_bots(store: PaperStore) -> list[dict[str, Any]]:
    bots_df = store.all_bots()
    bots = []
    if hasattr(bots_df, "iterrows"):
        for _, b in bots_df.iterrows():
            bots.append(
                {
                    "bot_id": str(b["bot_id"]),
                    "exchange": str(b["exchange"]),
                    "symbol": str(b["symbol"]),
                    "interval": str(b["interval"]),
                    "strategy": str(b["strategy"]),
                    "mode": str(b["mode"]),
                    "status": str(b["status"]),
                    "started_at": str(b["started_at"]),
                    "last_heartbeat": str(b["last_heartbeat"]),
                }
            )
    return bots


def _extract_trades(store: PaperStore, limit: int = 20) -> list[dict[str, Any]]:
    trades_df = store.all_trades()
    trades = []
    if hasattr(trades_df, "tail"):
        for _, t in trades_df.tail(limit).iloc[::-1].iterrows():
            trades.append(
                {
                    "ts": str(t.get("ts", "")),
                    "pair": str(t.get("pair", "")),
                    "symbol": str(t.get("symbol", "")),
                    "side": str(t.get("side", "")),
                    "size": float(t.get("size", 0.0)),
                    "entry_price": float(t.get("entry_price", 0.0) or 0.0),
                    "exit_price": float(t.get("exit_price", 0.0) or 0.0),
                    "pnl": float(t.get("pnl", 0.0) or 0.0),
                    "kind": str(t.get("kind", "")),
                }
            )
    return trades


async def sqlite_watcher() -> None:
    """Фонова задача: періодично перевіряє БД і розсилає WebSocket оновлення."""
    store = PaperStore()
    tick = 0
    while True:
        try:
            if manager.active_connections:
                tick += 1
                positions = _extract_positions(store)
                ctrl = load_control()

                # Кожну секунду: позиції та стан управління
                await manager.broadcast(
                    {
                        "type": "positions_update",
                        "data": positions,
                        "count": len(positions),
                        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                    }
                )

                # Кожні 2 секунди: акаунти та боти
                if tick % 2 == 0:
                    accounts = _extract_accounts(store)
                    bots = _extract_bots(store)
                    await manager.broadcast(
                        {
                            "type": "telemetry_update",
                            "data": {
                                "accounts": accounts,
                                "bots": bots,
                                "control": {
                                    "pause": ctrl.pause,
                                    "no_new_entries": ctrl.no_new_entries,
                                    "flatten": ctrl.flatten,
                                },
                            },
                            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                        }
                    )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Помилка SQLite watcher: %s", e)

        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _broadcast_task
    _broadcast_task = asyncio.create_task(sqlite_watcher())
    logger.info("FastAPI сервер запущено, SQLite watcher ініціалізовано.")
    yield
    if _broadcast_task:
        _broadcast_task.cancel()
    logger.info("FastAPI сервер зупинено.")


app = FastAPI(title="scalper-hft API", version="1.0.0", lifespan=lifespan)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.api_cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/api/v1/status")
async def get_status(token_payload: dict[str, Any] = Depends(verify_token)) -> dict[str, Any]:
    """Перевірка здоров'я системи та статистика."""
    settings = get_settings()
    ctrl = load_control()
    return {
        "status": "ok",
        "message": "scalper-hft API is running",
        "exchange": settings.exchange,
        "dry_run": settings.dry_run,
        "active_ws_connections": len(manager.active_connections),
        "control": {
            "pause": ctrl.pause,
            "no_new_entries": ctrl.no_new_entries,
            "flatten": ctrl.flatten,
        },
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    }


@app.get("/api/v1/control")
async def get_control(token_payload: dict[str, Any] = Depends(verify_token)) -> dict[str, Any]:
    """Отримати поточний стан керування ботами (control.json)."""
    ctrl = load_control()
    return {
        "pause": ctrl.pause,
        "no_new_entries": ctrl.no_new_entries,
        "flatten": ctrl.flatten,
    }


@app.post("/api/v1/control")
async def update_control(
    req: ControlUpdateRequest,
    token_payload: dict[str, Any] = Depends(verify_token),
) -> dict[str, Any]:
    """Оновити стан керування ботами (control.json)."""
    ctrl = save_control(
        pause=req.pause,
        no_new_entries=req.no_new_entries,
        flatten=req.flatten,
    )
    # Миттєво сповіщаємо клієнтів через WS
    await manager.broadcast(
        {
            "type": "control_update",
            "data": {
                "pause": ctrl.pause,
                "no_new_entries": ctrl.no_new_entries,
                "flatten": ctrl.flatten,
            },
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )
    return {
        "status": "ok",
        "control": {
            "pause": ctrl.pause,
            "no_new_entries": ctrl.no_new_entries,
            "flatten": ctrl.flatten,
        },
    }


@app.get("/api/v1/positions")
async def get_positions(token_payload: dict[str, Any] = Depends(verify_token)) -> dict[str, Any]:
    """Отримати поточні відкриті позиції."""
    store = PaperStore()
    positions = _extract_positions(store)
    return {"status": "ok", "positions": positions, "count": len(positions)}


@app.get("/api/v1/accounts")
async def get_accounts(token_payload: dict[str, Any] = Depends(verify_token)) -> dict[str, Any]:
    """Отримати останні баланси рахунків по біржах."""
    store = PaperStore()
    accounts = _extract_accounts(store)
    return {"status": "ok", "accounts": accounts, "count": len(accounts)}


@app.get("/api/v1/bots")
async def get_bots(token_payload: dict[str, Any] = Depends(verify_token)) -> dict[str, Any]:
    """Отримати список зареєстрованих ботів."""
    store = PaperStore()
    bots = _extract_bots(store)
    return {"status": "ok", "bots": bots, "count": len(bots)}


@app.get("/api/v1/trades")
async def get_trades(
    limit: int = 50,
    token_payload: dict[str, Any] = Depends(verify_token),
) -> dict[str, Any]:
    """Отримати останні виконані угоди."""
    store = PaperStore()
    trades = _extract_trades(store, limit=limit)
    return {"status": "ok", "trades": trades, "count": len(trades)}


@app.post("/api/v1/positions/close")
async def close_position(
    req: ClosePositionRequest,
    token_payload: dict[str, Any] = Depends(require_api_capability(destructive=True)),
) -> dict[str, Any]:
    """Закрити відкриту позицію (встановлює size=0)."""
    store = PaperStore()
    store.log_position(
        exchange=req.exchange,
        symbol=req.symbol,
        side="flat",
        size=0.0,
        entry_price=0.0,
        mark_price=0.0,
        unrealized_pnl=0.0,
        mode=req.mode,
    )
    # Миттєво розсилаємо оновлення
    positions = _extract_positions(store)
    await manager.broadcast(
        {
            "type": "positions_update",
            "data": positions,
            "count": len(positions),
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )
    return {"status": "ok", "message": f"Позицію {req.symbol} закрито"}


@app.post("/api/v1/emergency/flatten")
async def emergency_flatten(
    token_payload: dict[str, Any] = Depends(require_api_capability(destructive=True)),
) -> dict[str, Any]:
    """Аварійний Kill-Switch: встановлює flatten=True в control.json та закриває всі відкриті позиції в сховищі."""
    ctrl = save_control(pause=True, flatten=True)
    store = PaperStore()
    pos_list = store.open_positions()
    closed = 0
    if hasattr(pos_list, "iterrows"):
        for _, p in pos_list.iterrows():
            store.log_position(
                exchange=p["exchange"],
                symbol=p["symbol"],
                side="flat",
                size=0.0,
                entry_price=0.0,
                mark_price=0.0,
                unrealized_pnl=0.0,
                mode=p["mode"],
            )
            closed += 1

    positions = _extract_positions(store)
    await manager.broadcast(
        {
            "type": "positions_update",
            "data": positions,
            "count": len(positions),
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )
    await manager.broadcast(
        {
            "type": "control_update",
            "data": {
                "pause": ctrl.pause,
                "no_new_entries": ctrl.no_new_entries,
                "flatten": ctrl.flatten,
            },
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )
    return {
        "status": "ok",
        "message": f"Аварійний Kill-Switch активовано: закрито {closed} позицій, торгівлю призупинено.",
        "control": {"pause": ctrl.pause, "flatten": ctrl.flatten},
    }


@app.post("/api/v1/paper/mock_position")
async def create_mock_position(
    req: MockPositionRequest,
    token_payload: dict[str, Any] = Depends(require_api_capability(mock_positions=True)),
) -> dict[str, Any]:
    """Створити або оновити тестову paper-позицію для верифікації WebSocket зв'язку."""
    store = PaperStore()
    store.log_position(
        exchange=req.exchange,
        symbol=req.symbol,
        side=req.side,
        size=req.size,
        entry_price=req.entry_price,
        mark_price=req.entry_price * (1.0 + (0.01 if req.side == "long" else -0.01)),
        unrealized_pnl=req.unrealized_pnl,
        mode=req.mode,
    )
    positions = _extract_positions(store)
    await manager.broadcast(
        {
            "type": "positions_update",
            "data": positions,
            "count": len(positions),
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )
    return {"status": "ok", "message": f"Створено тестову позицію {req.symbol}", "data": req.model_dump()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket для субсекундних реал-тайм оновлень."""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return

    from fastapi.security import HTTPAuthorizationCredentials

    try:
        verify_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    except Exception:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket)
    store = PaperStore()
    # Відправляємо початковий знімок даних одразу при підключенні
    try:
        ctrl = load_control()
        initial_payload = {
            "type": "snapshot",
            "data": {
                "positions": _extract_positions(store),
                "accounts": _extract_accounts(store),
                "bots": _extract_bots(store),
                "control": {
                    "pause": ctrl.pause,
                    "no_new_entries": ctrl.no_new_entries,
                    "flatten": ctrl.flatten,
                },
            },
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        await websocket.send_json(initial_payload)
    except Exception as e:
        logger.warning("Не вдалося відправити initial snapshot у WS: %s", e)

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            elif data.startswith("{"):
                # Можливість обробляти команди від клієнта через WS
                try:
                    import json

                    cmd = json.loads(data)
                    if cmd.get("action") == "ping":
                        await websocket.send_json(
                            {
                                "type": "pong",
                                "echo_ts": cmd.get("ts"),
                                "server_ts": time.time(),
                            }
                        )
                except Exception:
                    pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
