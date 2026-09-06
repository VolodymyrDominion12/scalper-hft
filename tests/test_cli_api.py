"""CLI `api start --host/--port` parsing and bind overrides."""

from types import SimpleNamespace

import pytest
from scalper_hft.cli import _api_bind, main


@pytest.mark.parametrize(
    ("host", "port", "expected"),
    [
        (None, None, ("0.0.0.0", 8000)),
        ("127.0.0.1", None, ("127.0.0.1", 8000)),
        (None, 8080, ("0.0.0.0", 8080)),
        ("127.0.0.1", 8080, ("127.0.0.1", 8080)),
        ("", 8080, ("0.0.0.0", 8080)),
    ],
)
def test_api_bind_cli_overrides_defaults(host: str | None, port: int | None, expected: tuple[str, int]) -> None:
    assert _api_bind(host, port, default_host="0.0.0.0", default_port=8000) == expected


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_api_bind_rejects_invalid_port(port: int) -> None:
    with pytest.raises(ValueError, match="1-65535"):
        _api_bind("127.0.0.1", port, default_host="0.0.0.0", default_port=8000)


def test_api_start_parses_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_cmd(args: SimpleNamespace) -> None:
        captured["host"] = args.host
        captured["port"] = args.port

    monkeypatch.setattr("scalper_hft.cli.cmd_api", fake_cmd)
    main(["api", "start", "--host", "127.0.0.1", "--port", "8080"])
    assert captured == {"host": "127.0.0.1", "port": 8080}


def test_api_start_port_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_cmd(args: SimpleNamespace) -> None:
        captured["host"] = args.host
        captured["port"] = args.port

    monkeypatch.setattr("scalper_hft.cli.cmd_api", fake_cmd)
    main(["api", "start"])
    assert captured == {"host": None, "port": None}
