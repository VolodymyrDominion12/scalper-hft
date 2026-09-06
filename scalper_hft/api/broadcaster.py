"""WebSocket ConnectionManager для реал-тайм розсилки оновлень (Push)."""

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Управляє активними WebSocket з'єднаннями."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Клієнт підключився до WS. Всього підключень: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Клієнт відключився від WS. Всього підключень: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Надіслати JSON повідомлення всім підключеним клієнтам."""
        if not self.active_connections:
            return
        
        payload = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except WebSocketDisconnect:
                self.disconnect(connection)
            except Exception as e:
                logger.error(f"Помилка відправки WS повідомлення: {e}")
                self.disconnect(connection)


manager = ConnectionManager()
