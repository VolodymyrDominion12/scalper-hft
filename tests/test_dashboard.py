"""Дашборд: sys.path під Streamlit і CLI-підкоманда."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]


def test_cli_job_help() -> None:
    from scalper_hft.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["job", "--help"])
    assert ei.value.code == 0


def test_sidebar_payload_fingerprint_stable() -> None:
    from scalper_hft.research.jobs import fingerprint

    payload = {
        "strategy": "pairs_arb",
        "leg1": "XRPUSDT",
        "leg2": "BTCUSDT",
        "interval": "1h",
        "days": 90,
        "params": {},
        "maker": True,
        "position_pct": 0.3,
        "base_interval": "1m",
    }
    assert fingerprint("pairs", payload) == fingerprint("pairs", dict(payload))
    assert fingerprint("pairs", payload) != fingerprint("pairs", {**payload, "days": 30})


def test_streamlit_style_path_can_import_package() -> None:
    """Як Anaconda+Streamlit: на path лише каталог скрипта, без site-packages пакету."""
    code = r"""
import sys
from pathlib import Path

repo = Path(".").resolve()
script_dir = repo / "scalper_hft"
kept: list[str] = []
for p in sys.path:
    if p in ("", "."):
        continue
    rp = Path(p).resolve()
    s = rp.as_posix().lower()
    if rp == repo or rp == script_dir:
        continue
    if "site-packages" in s or "dist-packages" in s:
        continue
    if "scalper" in s:
        continue
    kept.append(p)
sys.path = [str(script_dir), *kept]
for k in list(sys.modules):
    if k == "scalper_hft" or k.startswith("scalper_hft."):
        del sys.modules[k]

try:
    import scalper_hft  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    raise SystemExit("expected ModuleNotFoundError before bootstrap")

sys.path.insert(0, str(repo))
from scalper_hft.app_pages._common import SYMBOLS

assert "BTCUSDT" in SYMBOLS
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout


def test_reload_shared_restores_stale_common_names() -> None:
    """Імітація Streamlit: у sys.modules лежить старий _common без нових імен."""
    import scalper_hft.app_pages._common as common
    from scalper_hft.app_pages import reload_shared

    assert hasattr(common, "RESEARCH_SECTION")
    del common.RESEARCH_SECTION
    del common.JOB_KINDS
    assert not hasattr(common, "RESEARCH_SECTION")
    reload_shared()
    import scalper_hft.app_pages._common as refreshed

    assert refreshed.RESEARCH_SECTION == "research_section_prefill"
    assert "capacity" in refreshed.JOB_KINDS


def test_settings_has_dashboard_password_hash() -> None:
    from scalper_hft.config import Settings

    settings = Settings(dashboard_password_hash=" $2b$12$test ")
    assert settings.dashboard_password_hash == " $2b$12$test "


def test_get_settings_replaces_stale_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Як після Streamlit-reload: у `_settings` лежить об'єкт без нових полів."""
    import scalper_hft.config as cfg

    monkeypatch.setattr(cfg, "_settings", SimpleNamespace(dry_run=True))
    settings = cfg.get_settings()
    assert isinstance(settings, cfg.Settings)
    assert hasattr(settings, "dashboard_password_hash")


def test_stored_password_hash_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from scalper_hft.dashboard_auth import stored_password_hash

    monkeypatch.setattr(
        "scalper_hft.dashboard_auth.get_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.delenv("DASHBOARD_PASSWORD_HASH", raising=False)
    assert stored_password_hash() == ""


def test_stored_password_hash_env_fallback_when_settings_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Регресія AttributeError: старий Settings без dashboard_password_hash."""
    from scalper_hft.dashboard_auth import stored_password_hash

    monkeypatch.setattr(
        "scalper_hft.dashboard_auth.get_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", "  $2b$12$examplehash  ")
    assert stored_password_hash() == "$2b$12$examplehash"


def test_check_password_skips_auth_when_hash_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    from scalper_hft.dashboard_auth import check_password

    monkeypatch.setattr("scalper_hft.dashboard_auth.stored_password_hash", lambda: "")
    assert check_password() is True


def test_generate_hash_roundtrip() -> None:
    import bcrypt
    from scalper_hft.dashboard_auth import generate_hash

    hashed = generate_hash("secret-pass")
    assert bcrypt.checkpw(b"secret-pass", hashed.encode("utf-8"))
    assert not bcrypt.checkpw(b"wrong", hashed.encode("utf-8"))
