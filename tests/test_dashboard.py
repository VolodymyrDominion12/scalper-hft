"""Дашборд: sys.path під Streamlit і CLI-підкоманда."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
