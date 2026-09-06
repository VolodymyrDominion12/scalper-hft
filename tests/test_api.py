"""Тести для FastAPI сервера, розширених REST ендпоінтів та WebSocket."""

import pytest
from fastapi.testclient import TestClient
from scalper_hft.api.auth import create_access_token
from scalper_hft.api.server import app
from scalper_hft.live.control import load_control

client = TestClient(app)


@pytest.fixture
def auth_token() -> str:
    """Генерує валідний тестовий токен."""
    return create_access_token({"sub": "test_user"}, expires_delta_hours=1)


def test_status_unauthorized():
    response = client.get("/api/v1/status")
    assert response.status_code in (401, 403)


def test_status_authorized(auth_token: str):
    response = client.get(
        "/api/v1/status",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "active_ws_connections" in data
    assert "control" in data


def test_control_endpoints(auth_token: str):
    headers = {"Authorization": f"Bearer {auth_token}"}

    # 1. GET /api/v1/control
    get_res = client.get("/api/v1/control", headers=headers)
    assert get_res.status_code == 200
    ctrl_data = get_res.json()
    assert "pause" in ctrl_data

    # 2. POST /api/v1/control (Pause)
    post_res = client.post(
        "/api/v1/control",
        headers=headers,
        json={"pause": True, "no_new_entries": False},
    )
    assert post_res.status_code == 200
    assert post_res.json()["control"]["pause"] is True
    assert load_control().pause is True

    # 3. Resume
    post_res2 = client.post(
        "/api/v1/control",
        headers=headers,
        json={"pause": False, "flatten": False},
    )
    assert post_res2.status_code == 200
    assert post_res2.json()["control"]["pause"] is False
    assert load_control().pause is False


def test_accounts_endpoint(auth_token: str):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.get("/api/v1/accounts", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "accounts" in data


def test_bots_endpoint(auth_token: str):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.get("/api/v1/bots", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "bots" in data


def test_trades_endpoint(auth_token: str):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.get("/api/v1/trades?limit=10", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "trades" in data


def test_mock_and_close_position(auth_token: str):
    headers = {"Authorization": f"Bearer {auth_token}"}

    # 1. Створюємо mock-позицію
    mock_res = client.post(
        "/api/v1/paper/mock_position",
        headers=headers,
        json={
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.5,
            "entry_price": 60000.0,
            "unrealized_pnl": 50.0,
            "exchange": "binance",
            "mode": "paper",
        },
    )
    assert mock_res.status_code == 200
    assert mock_res.json()["status"] == "ok"

    # 2. Отримуємо відкриті позиції
    pos_res = client.get("/api/v1/positions", headers=headers)
    assert pos_res.status_code == 200
    positions = pos_res.json()["positions"]
    btc_pos = [p for p in positions if p["symbol"] == "BTCUSDT"]
    assert len(btc_pos) >= 1
    assert btc_pos[0]["size"] == 0.5

    # 3. Закриваємо позицію
    close_res = client.post(
        "/api/v1/positions/close",
        headers=headers,
        json={"symbol": "BTCUSDT", "exchange": "binance", "mode": "paper"},
    )
    assert close_res.status_code == 200

    # 4. Перевіряємо що позиція закрита (size == 0 більше не потрапляє в open_positions)
    pos_res2 = client.get("/api/v1/positions", headers=headers)
    assert pos_res2.status_code == 200
    positions2 = pos_res2.json()["positions"]
    btc_pos2 = [p for p in positions2 if p["symbol"] == "BTCUSDT"]
    assert len(btc_pos2) == 0


def test_emergency_flatten(auth_token: str):
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Створюємо тестову позицію
    client.post(
        "/api/v1/paper/mock_position",
        headers=headers,
        json={
            "symbol": "ETHUSDT",
            "side": "short",
            "size": 2.0,
            "entry_price": 3000.0,
            "unrealized_pnl": -20.0,
            "exchange": "binance",
            "mode": "paper",
        },
    )

    # Викликаємо аварійний flatten
    flatten_res = client.post("/api/v1/emergency/flatten", headers=headers)
    assert flatten_res.status_code == 200
    assert flatten_res.json()["status"] == "ok"
    assert load_control().flatten is True

    # Перевіряємо що позиції закриті
    pos_res = client.get("/api/v1/positions", headers=headers)
    assert pos_res.status_code == 200
    positions = pos_res.json()["positions"]
    eth_pos = [p for p in positions if p["symbol"] == "ETHUSDT"]
    assert len(eth_pos) == 0

    # Відновлюємо нормальний стан для наступних тестів
    client.post(
        "/api/v1/control",
        headers=headers,
        json={"pause": False, "no_new_entries": False, "flatten": False},
    )


def test_websocket_unauthorized():
    with pytest.raises(Exception):
        with client.websocket_connect("/ws"):
            pass


def test_websocket_authorized_and_snapshot(auth_token: str):
    with client.websocket_connect(f"/ws?token={auth_token}") as websocket:
        # Сервер одразу надсилає initial snapshot
        initial_data = websocket.receive_json()
        assert initial_data["type"] == "snapshot"
        assert "positions" in initial_data["data"]

        # Тест ping -> pong
        websocket.send_text("ping")
        data = websocket.receive_text()
        assert data == "pong"
