"""W0-R5: синтаксис і dry-run шлях scripts/sync_depth.sh (без SSH)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "sync_depth.sh"


def test_sync_depth_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(_SCRIPT)], check=True)


def test_sync_depth_requires_source() -> None:
    env = os.environ.copy()
    env.pop("VPS_HOST", None)
    r = subprocess.run(["bash", str(_SCRIPT)], cwd=_REPO, env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "VPS_HOST" in r.stderr or "user@host" in r.stderr


def test_sync_depth_help() -> None:
    r = subprocess.run(["bash", str(_SCRIPT), "--help"], cwd=_REPO, capture_output=True, text=True)
    assert r.returncode == 0
    assert "--dry-run" in r.stderr


def test_sync_depth_dry_run_skips_audit(tmp_path: Path) -> None:
    fake = tmp_path / "bin"
    fake.mkdir()
    rsync = fake / "rsync"
    rsync.write_text('#!/bin/sh\necho rsync-stub "$@"\nexit 0\n', encoding="utf-8")
    rsync.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake}{os.pathsep}{env.get('PATH', '')}"
    env["LOCAL_DIR"] = str(tmp_path / "data")
    env["QUALITY_OUT"] = str(tmp_path / "quality_depth.md")
    env.pop("VPS_HOST", None)
    r = subprocess.run(
        ["bash", str(_SCRIPT), "--dry-run", "tradebot@example:~/scalper-hft/data"],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    combined = r.stdout + r.stderr
    assert "--dry-run" in combined
    assert "--partial" in combined
    assert "--whole-file" in combined
    assert "пропуск validate" in combined
    assert not Path(env["QUALITY_OUT"]).exists()
