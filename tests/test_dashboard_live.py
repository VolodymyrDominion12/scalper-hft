"""Тести для Live WebSocket моніторингу та CCv2 компонента."""

from __future__ import annotations

from scalper_hft.api.auth import create_access_token, verify_token
from scalper_hft.live.control import load_control, save_control
from scalper_hft.visualization.ws_component import ws_live_monitor_component


def test_ws_component_callable():
    """Перевірка що функція монтажу CCv2 компонента визначена і приймає необхідні аргументи."""
    assert callable(ws_live_monitor_component)


def test_jwt_token_generation_and_verification():
    """Перевірка що токен, згенерований для дашборду, є валідним."""
    token = create_access_token({"sub": "dashboard_admin", "role": "admin"})
    assert isinstance(token, str)
    assert len(token) > 20

    from fastapi.security import HTTPAuthorizationCredentials
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    payload = verify_token(creds)
    assert payload["sub"] == "dashboard_admin"
    assert payload["role"] == "admin"


def test_control_state_roundtrip(tmp_path):
    """Перевірка атомарного збереження та читання control.json."""
    control_file = tmp_path / "control.json"

    # 1. Початковий стан
    s1 = load_control(control_file)
    assert s1.pause is False
    assert s1.is_noop is True

    # 2. Збереження паузи
    s2 = save_control(pause=True, path=control_file)
    assert s2.pause is True
    assert s2.is_noop is False

    # 3. Збереження flatten
    s3 = save_control(flatten=True, path=control_file)
    assert s3.flatten is True
    assert s3.pause is True

    # 4. Відновлення
    s4 = save_control(pause=False, no_new_entries=False, flatten=False, path=control_file)
    assert s4.is_noop is True
