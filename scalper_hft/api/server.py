"""FastAPI сервер: REST API та WebSocket для дашборду."""

import asyncio
import logging
from typing import Any
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from scalper_hft.api.auth import verify_token
from scalper_hft.api.broadcaster import manager
from scalper_hft.live.store import PaperStore

logger = logging.getLogger(__name__)

# Фонова задача для broadcasting
_broadcast_task: asyncio.Task | None = None

async def sqlite_watcher() -> None:
    """Фонова задача: періодично перевіряє БД і розсилає оновлення.
    Оскільки SQLite (у WAL режимі) може бути змінений іншим процесом,
    ми робимо polling раз на секунду і розсилаємо підключеним клієнтам.
    """
    store = PaperStore()
    while True:
        try:
            if manager.active_connections:
                pos_list = store.open_positions()
                positions = []
                for _, p in pos_list.iterrows() if hasattr(pos_list, "iterrows") else enumerate(pos_list):
                    positions.append({
                        "symbol": p["symbol"] if hasattr(p, "__getitem__") else p.symbol,
                        "side": p["side"] if hasattr(p, "__getitem__") else p.side,
                        "size": p["size"] if hasattr(p, "__getitem__") else p.size,
                        "entry_price": p["entry_price"] if hasattr(p, "__getitem__") else p.entry_price,
                        "unrealized_pnl": p["unrealized_pnl"] if hasattr(p, "__getitem__") else p.unrealized_pnl,
                        "ts": p["ts"] if hasattr(p, "__getitem__") else p.ts,
                    })
                
                await manager.broadcast({
                    "type": "positions_update",
                    "data": positions
                })
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Помилка SQLite watcher: {e}")
        
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/status")
async def get_status(token_payload: dict[str, Any] = Depends(verify_token)) -> dict[str, Any]:
    """Перевірка здоров'я системи."""
    return {"status": "ok", "message": "scalper-hft API is running"}


@app.get("/api/v1/positions")
async def get_positions(token_payload: dict[str, Any] = Depends(verify_token)) -> dict[str, Any]:
    """Отримати поточні відкриті позиції."""
    store = PaperStore()
    pos_list = store.open_positions()
    positions = []
    for _, p in pos_list.iterrows() if hasattr(pos_list, "iterrows") else enumerate(pos_list):
        positions.append({
            "id": p["id"] if hasattr(p, "__getitem__") else getattr(p, "id", None),
            "symbol": p["symbol"] if hasattr(p, "__getitem__") else p.symbol,
            "side": p["side"] if hasattr(p, "__getitem__") else p.side,
            "size": p["size"] if hasattr(p, "__getitem__") else getattr(p, "size", 0.0),
            "entry_price": p["entry_price"] if hasattr(p, "__getitem__") else getattr(p, "entry_price", 0.0),
            "unrealized_pnl": p["unrealized_pnl"] if hasattr(p, "__getitem__") else getattr(p, "unrealized_pnl", 0.0),
            "ts": p["ts"] if hasattr(p, "__getitem__") else getattr(p, "ts", ""),
        })
    return {"status": "ok", "positions": positions}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket для реал-тайм оновлень."""
    # Оскільки WS підключення може не мати заголовка Authorization у деяких браузерах,
    # ми отримуємо token через query parameter `?token=...`
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
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
