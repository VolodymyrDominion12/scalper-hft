"""Тести для FastAPI сервера та WebSocket."""

import pytest
from fastapi.testclient import TestClient

from scalper_hft.api.auth import create_access_token
from scalper_hft.api.server import app

client = TestClient(app)

@pytest.fixture
def auth_token() -> str:
    """Генерує валідний тестовий токен."""
    return create_access_token({"sub": "test_user"}, expires_delta_hours=1)

def test_status_unauthorized():
    response = client.get("/api/v1/status")
    assert response.status_code == 403 or response.status_code == 401

def test_status_authorized(auth_token: str):
    response = client.get(
        "/api/v1/status",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "scalper-hft API is running"}

def test_positions_authorized(auth_token: str):
    response = client.get(
        "/api/v1/positions",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    assert "positions" in response.json()

def test_websocket_unauthorized():
    with pytest.raises(Exception):
        with client.websocket_connect("/ws"):
            pass

def test_websocket_authorized(auth_token: str):
    with client.websocket_connect(f"/ws?token={auth_token}") as websocket:
        websocket.send_text("ping")
        data = websocket.receive_text()
        assert data == "pong"
